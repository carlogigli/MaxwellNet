# Copyright (c) 2021 Joowon Lim - Carlo Gigli

import torch
from torch import nn
from TUNet import TUNet
import torch.nn.functional as F

import math
import numpy as np
import matplotlib.pyplot as plt
import constants
from utils import _complex_multiplication, _complex_scalar_multiplication

epsilon0 = 8.85e-12
mu0 = 4 * math.pi * 1e-7
eta0 = np.sqrt(mu0 / epsilon0)


class MaxwellNet(nn.Module):
    """ Physical driven network base class.
        It inheritates the properties of Module class in PyTorch and initialize a subclass (TUNet) defyning the core neural network
    """
    def __init__(self, depth=5, filter=6, norm='weight', up_mode='upconv',
                 wavelength=0.5, nb = 1, n2=0, dpl=16, Nx=256, Nz=256, pml_thickness=16, symmetry_x=False, periodic=False, loc_source=False, mode='te', high_order=True):
        """
        Parameters
        ----------
        depth : int, default 5
            U-Net module depth
        filter : int, default 6
            Number of channels in first layer of U-Net encoder
        norm : string, default 'weight'
            Normalization used in U-Net ('weight', 'batch', 'none')
        up_mode : string, default 'upconv'
            Upscaling mode used in U-Net decoder (upconv, upsample)
        wavelength : float, default 0.5
            Working wavelength **in micron**. *TODO: change this in SI units*
        nb : float, default 1.0
            Background material refractive index
        n2 : float, default 0.
            Nonlinear refractive index in m2/W
        dpl : int, default 16
            Dots-per-wavelength to define Yee grid. The spatial resolution will be wavelength/dpl
        Nx : int, default 256
            Number of pixels along x *TODO: change the coordinate system. Let x be the horizontal axis and y the vertical one*
        Nz : int, default 256
            Number of pixels along z
        pml_thickness : int, default 16
            thickness in pixels of PML region. Usually in FDFD simulations this is of the order of 10/20.
            Check the parameters in set_pml_tensors method to tune the pml
        symmetry_x : bool, default False
            Symmetry along x axis (True/False).
        mode : string, default 'te'
            Incident wave mode ('te' or 'tm'). The used convention is 'te' out of plane polarization (Ey the only non zero
            component). **This is the opposite convention used in the photonic crytal community**
        high_order: bool, default True
            Finite differences discretization order. If False it used second order discretization, if True it used
            fourth order. See http://www.math.umassd.edu/~cwang/FWWY08.pdf
        """

        super(MaxwellNet, self).__init__()
        self.mode = mode

        in_channels = 1 # Refractive index distribution
        groups = 1 # 1
        if mode == 'te':
            out_channels = 2 # Ey field real and complex parts
        elif mode == 'tm':
            out_channels = 4 # Ex + Ez fields real and complex parts

        self.high_order = high_order
        self.pml_thickness = pml_thickness
        # Initialization of tunable U-Net module
        self.model = TUNet(in_channels, out_channels, depth, filter, norm, up_mode, groups)

        # pixel size [um / pixel]
        delta = wavelength / dpl / nb
        # wave-number [1 / um]
        k = 2 * math.pi / wavelength
        self.delta = delta
        self.k = k
        self.omega = constants.C0 * k
        self.nb = nb
        self.n2 = n2

        self.symmetry_x = symmetry_x
        self.periodic = periodic

        if self.high_order == 'second':
            pad = 2
            self.pad = pad
        elif self.high_order == 'fourth':
            pad = 4
            self.pad = pad

        # Define paddings methods for convolutions
        self.padding_ref = nn.Sequential(nn.ReflectionPad2d((0, 0, pad, 0)), nn.ZeroPad2d((pad, pad, 0, pad)))
        self.padding_zero = nn.Sequential(nn.ZeroPad2d((pad, pad, pad, pad)))
        self.padding_zero_x = nn.Sequential(nn.ZeroPad2d((0, 0, pad // 2, pad // 2)))
        self.padding_zero_z = nn.Sequential(nn.ZeroPad2d((pad // 2, pad // 2, 0, 0)))
        self.padding_replication = nn.Sequential(nn.ReplicationPad2d(1))

        if symmetry_x == True:
            Nx = Nx // (symmetry_x + 1)
            x = np.linspace(-pad, Nx + pad - 1, Nx + 2 * pad) * delta
        else:
            x = np.linspace(-Nx // 2 - pad, Nx // 2 + pad - 1, Nx + 2 * pad) * delta
        z = np.linspace(-Nz // 2 - pad, Nz // 2 + pad - 1, Nz + 2 * pad) * delta

        # Coordinate set-up
        zz, xx = np.meshgrid(z, x)
        self.Nx = zz.shape[0]
        self.Nz = zz.shape[1]
        self.x = x
        self.z = z
        self.zz = zz
        self.xx = xx

        # Fast oscillating term of background electric and magnetic fields on the Yee grid
        # This corresponds to the input field propagating along z-axis in a uniform medium with refractive index nb

        eb = np.exp(1j * (k * nb * zz))
        self.register_buffer('eb', complex_tensor(eb))

        if self.periodic:
            self.set_pml_tensors(direction='z')
        else:
            self.set_pml_tensors(direction='xz')

        if loc_source: # Initialize a null local source
            self.insert_localized_source(np.zeros((Nx,Nz), dtype=complex))

        self.set_gradient_kernels()

        # Compute second z-derivative incident field
        self.register_buffer('dd_z_eb', (self.dd_z(complex_tensor(np.exp(1j * (k * nb * zz))))))

        # Define kernel for smoothing input index distributions
        ker_smooth = torch.tensor([[0, 1 / 8, 0], [1 / 8, 1 / 2, 1 / 8], [0, 1 / 8, 0]], dtype=torch.float32,
                                  requires_grad=False)
        self.register_buffer('conv_smooth', torch.zeros((1, 1, 3, 3), dtype=torch.float32, requires_grad=False))
        self.conv_smooth[0, 0, :, :] = ker_smooth

    def forward(self, ref_index, S0):
        """Forward propagation. It takes as an input the discretized refractive index distribution [Nz x Nx x 1] and the incident intensity.
        It returns the total field (scattered + incident)

        Parameters
        ----------
        ref_index : PyTorch tensor with size (Nb, 1, Nx, Nz)
            Refractive index distribution. Eventually containing a batch of size Nb
        S0 : PyTorch tensor with size (Nb,1)
            Incident plane wave intensity

        Return
        -------
        total field : PyTorch tensor with size (Nb, 2, Nx, Nz)
            Total field related to the Nb refractive index distributions. The two channels contain real and imaginary parts, respectively.
        """

        tensor_zero = torch.zeros((ref_index.shape[0], 2, self.Nx-2*self.pad, self.Nz-2*self.pad)).to(ref_index.get_device())
        scattered = {'Ex':complex_tensor(tensor_zero), 'Ey':complex_tensor(tensor_zero), 'Ez':complex_tensor(tensor_zero)}
        total = {'Ex':complex_tensor(tensor_zero), 'Ey':complex_tensor(tensor_zero), 'Ez':complex_tensor(tensor_zero)}

        if self.mode == 'te':

            scattered['Ey'] = complex_tensor(self.model(ref_index, S0)[0]) #Scattered field
            total['Ey'] = scattered['Ey'] + self.nopad(self.eb)

        elif self.mode == 'tm':

            out, _ = self.model(ref_index, S0)  # Scattered field
            scattered['Ex'] = complex_tensor(out[:, 0:2, :, :])
            scattered['Ez'] = complex_tensor(out[:, 2:4, :, :])
            total['Ex'] = scattered['Ex'] + self.nopad(self.eb)
            total['Ez'] = scattered['Ez']


        if hasattr(self, 'j_src'):
            return scattered
        else:
            return total

    def loss(self, total, ref_index, S0):
        """It computes the physical loss based on Maxwell's equation residual. It takes as an input the
        discretized refractive index distribution [Nz x Nx x 1], the incident intensity and the total field computed with
        forward method.

        Parameters
        ----------
        total : PyTorch tensor with size (Nb, 2, Nx, Nz)
            total field predicted by forward method
        ref_index : PyTorch tensor with size (Nb, 1, Nx, Nz)
        S0 : PyTorch tensor with size (Nb,1)

        Return
        ------
        loss : PyTorch tensor with size (Nb, 2, Nx, Nz)
            Numerically evaluated loss, point by point on the Yee grid.
        """

        E0 = torch.sqrt(2 * constants.Z0 / self.nb * S0).unsqueeze(1).unsqueeze(2).unsqueeze(3)

        if self.mode == 'te':

            ey = total['Ey']
            I = E0 ** 2 * ref_index * (ey * ey.conj()).real() / (2 * constants.Z0)
            if hasattr(self.n2, "__len__"):
                n = ref_index + torch.from_numpy(self.n2).unsqueeze(0).unsqueeze(1).to(I.get_device()) * I * (ref_index > self.nb)
            else:
                n = ref_index + self.n2 * I * (ref_index > self.nb)
            epsilon = (n ** 2).float()
            epsilon = self.padding_replication(F.conv2d(epsilon, self.conv_smooth, padding=0, groups=1))

            if hasattr(self, 'j_src'):

                if self.symmetry_x == True:
                    ey = self.padding_ref(ey)
                else:
                    ey = self.padding_zero(ey)

                diff = self.nopad(self.dd_x_pml(ey) + self.dd_z_pml(ey)) \
                       + self.k ** 2 * (epsilon * self.nopad(ey)) \
                       + self.k * constants.Z0 * self.complex_scalar_multiplication(self.j_src, 1j)

            else:

                ey_i = self.nopad(self.eb)
                ey_s = ey - ey_i

                if self.symmetry_x == True:
                    ey_s = self.padding_ref(ey_s)
                elif self.periodic:
                    ey = self.padding_pbc(ey, inc_scat='inc')
                    ey_s = self.padding_pbc(ey_s, inc_scat='scat')
                else:
                    ey_s = self.padding_zero(ey_s)

                diff = self.nopad(self.dd_x_pml(ey_s) + self.dd_z_pml(ey_s)) \
                       + self.nopad(self.dd_z_eb) \
                       + self.k ** 2 * (epsilon * ey) \

        elif self.mode == 'tm':

            ex = total['Ex']
            ex_i = self.nopad(self.eb)
            ex_s = ex - ex_i

            ez_s = total['Ez']

            I = E0 ** 2 * ref_index * ((ex * ex.conj()).real()+(ez_s * ez_s.conj()).real()) / (2 * constants.Z0)

            if hasattr(self.n2, "__len__"):
                n = ref_index + torch.from_numpy(self.n2).unsqueeze(0).unsqueeze(1).to(I.get_device()) * I * (
                            ref_index > self.nb)
            else:
                n = ref_index + self.n2 * I * (ref_index > self.nb)

            epsilon = (n ** 2).float()
            epsilon = self.padding_replication(F.conv2d(epsilon, self.conv_smooth, padding=0, groups=1))

            if self.symmetry_x:
                ex_s = self.padding_zero(ex_s)
                ez_s = self.padding_ref(ez_s)
                ex_s[:, :, 0:self.pad, :] = torch.flip(ex_s[:, :, self.pad:2 * self.pad, :], [2])
                ez_s[:, :, 0:self.pad, :] = -ez_s[:, :, 0:self.pad, :]
            elif self.periodic:
                ex = self.padding_pbc(ex, inc_scat='inc')
                ex_s = self.padding_pbc(ex_s, inc_scat='scat')
                ez_s = self.padding_pbc(ez_s, inc_scat='scat')

            else:
                ex_s = self.padding_zero(ex_s)
                ez_s = self.padding_zero(ez_s)

            diff_x = self.nopad(self.dd_z_pml(ex_s) - self.dd_zx_pml(ez_s)) \
                     + self.nopad(self.dd_z_eb) \
                     + self.k ** 2 * (epsilon * ex)

            diff_z = self.nopad(self.dd_x_pml(ez_s) - self.dd_xz_pml(ex_s)) \
                     + self.k ** 2 * (epsilon * self.nopad(ez_s))

            diff = torch.cat((diff_x, diff_z), 1)

        return diff

    def set_inhomogeneous_chi3(self, chi3):
        """

        Parameters
        ----------
        chi3

        Returns
        -------

        """
        self.n2 = self.n2*chi3
        return

    def set_gradient_kernels(self):
        """Setup convolutional kernels for finite differences method on Yee grid
        If self.high_order = second it implements simple middle point differences
        If self.high_order = fourth it implements fourth order difference scheme
        See: A.Fathy et al. “A fourth order difference scheme for the Maxwell equations on Yee grid,” J. Hyperbolic Differ. Equations 5, 613–642 (2008)
        """
        if self.pad == 2:
            ker_h = torch.tensor([-1, +1, 0], dtype=torch.float32, requires_grad=False)
            ker_e = torch.tensor([0, -1, +1], dtype=torch.float32, requires_grad=False)
        else:
            ker_h = torch.tensor([1 / 24, -9 / 8, +9 / 8, -1 / 24, 0], dtype=torch.float32, requires_grad=False)
            ker_e = torch.tensor([0, 1 / 24, -9 / 8, +9 / 8, -1 / 24], dtype=torch.float32, requires_grad=False)

        # Gradient and laplacian kernel set up for magnetic fields
        self.register_buffer('gradient_h_z', torch.zeros((2, 1, 1, self.pad + 1), dtype=torch.float32, requires_grad=False))
        self.register_buffer('gradient_h_x', torch.zeros((2, 1, self.pad + 1, 1), dtype=torch.float32, requires_grad=False))
        self.gradient_h_z[:, :, 0, :] = ker_h/self.delta
        self.gradient_h_x = self.gradient_h_z.permute(0, 1, 3, 2)

        # Gradient and laplacian kernel set up for electric fields (1 voxel shifted wrt magnetic fields)
        self.register_buffer('gradient_e_z',
                             torch.zeros((2, 1, 1, self.pad + 1), dtype=torch.float32, requires_grad=False))
        self.register_buffer('gradient_e_x',
                             torch.zeros((2, 1, self.pad + 1, 1), dtype=torch.float32, requires_grad=False))
        self.gradient_e_z[:, :, 0, :] = ker_e / self.delta
        self.gradient_e_x = self.gradient_e_z.permute(0, 1, 3, 2)

    def set_pml_tensors(self, direction = 'xz'):

        # perfectly matching layer set up
        m = 4
        const = 20

        if 'x' in direction:
            rx_p = 1 + 1j * const * (self.xx - self.x[-1] + self.pml_thickness * self.delta) ** m
            rx_p[0:-self.pml_thickness, :] = 0
            rx_n = 1 + 1j * const * (self.xx - self.x[0] - self.pml_thickness * self.delta) ** m
            rx_n[self.pml_thickness:, :] = 0
            rx = rx_p + rx_n
            if self.symmetry_x == True:
                rx[0:-self.pml_thickness, :] = 1
            else:
                rx[self.pml_thickness:-self.pml_thickness, :] = 1
        else:
            rx = np.ones_like(self.xx, dtype=complex)

        if 'z' in direction:
            rz_p = 1 + 1j * const * (self.zz - self.z[-1] + self.pml_thickness * self.delta) ** m
            rz_p[:, 0:-self.pml_thickness] = 0
            rz_n = 1 + 1j * const * (self.zz - self.z[0] - self.pml_thickness * self.delta) ** m
            rz_n[:, self.pml_thickness::] = 0
            rz = rz_p + rz_n
            rz[:, self.pml_thickness:-self.pml_thickness] = 1
        else:
            rz = np.ones_like(self.zz, dtype=complex)

        rx_inverse = 1 / rx
        rz_inverse = 1 / rz

        self.register_buffer('rx_inverse',complex_tensor(rx_inverse))
        self.register_buffer('rz_inverse', complex_tensor(rz_inverse))

    def padding_pbc(self, input, inc_scat='scat'):
        """
        Set periodic boundary conditions on top and bottom sides with periodic paddings.
        At the current stage it imposes Floquet-Bloch periodic conditions for normal incidence
        Etop=Ebot exp(-j kF Lx) with kF=0 -> Etop=Ebot

        """
        if inc_scat=='scat':
            output = F.pad(input[:, :, :-1, :], (0, 0, self.pad, self.pad + 1), mode='circular')
            output = F.pad(output, (self.pad, self.pad, 0, 0))
        elif inc_scat=='inc':
            output = F.pad(input[:, :, :-1, :], (0, 0, 0, 1), mode='circular')
        else:
            raise ValueError('Invalid incident/scattered field condition. Set inc_scat = inc or scat')
        return output

    def insert_localized_source(self, source):
        """
        It defines a localized source in the domain, e.g. a waveguide mode or a dipole. This term will be used as the source term
        in Maxwell's equation and the background plane wave excitation is removed.

        source: source current distribution
        coord: a dictionary containing z,x coordinates of source term
        """

        self.register_buffer('j_src',complex_tensor(torch.zeros([1, 2, self.Nx-2*self.pad, self.Nz-2*self.pad], dtype=torch.float32,requires_grad=False)))
        self.j_src[0, 0, :, :] = torch.from_numpy(np.real(source))
        self.j_src[0, 1, :, :] = torch.from_numpy(np.imag(source))

        return

    def nopad(self, tensor):
        """ It returns a tensor without the external padding

        Parameters
        ----------
        tensor : PyTorch Tensor of size (Nb, any, Nx, Nz)

        Returns
        -------
        PyTorch Tensor of size (Nb, any, Nx-2*pad, Nz-2*pad)
        """
        return tensor[:, :, self.pad:-self.pad, self.pad:-self.pad]

    def d_e_x(self, x):
        """
        Returns the first order x derivative on the Yee grid for the electric field

        Input
        ------------------------------
        x : PyTorch tensor with dimensions (N,C,H,W) where
            - N: number of groups (default = 1)
            - C: number of channels (=2 being real and imaginary parts)
            - H: height (x axis)
            - W: width (z axis)

        Output
        -----------------------------
        PyTorch tensor with dimensions (N,C,H,W)
        """
        return self.padding_zero_x(F.conv2d(x, self.gradient_e_x, padding=0, groups=2))

    def d_h_x(self, x):
        """
        Returns the first order x derivative on the Yee grid for the magnetic field

        Input
        ------------------------------
        x : PyTorch tensor with dimensions (N,C,H,W) where
            - N: number of groups (default = 1)
            - C: number of channels (=2 being real and imaginary parts)
            - H: height (x axis)
            - W: width (z axis)

        Output
        -----------------------------
        PyTorch tensor with dimensions (N,C,H,W)
        """
        return self.padding_zero_x(F.conv2d(x, self.gradient_h_x, padding=0, groups=2))

    def d_e_z(self, x):
        """
        Returns the first order z derivative on the Yee grid for the electric field

        Input
        ------------------------------
        x : PyTorch tensor with dimensions (N,C,H,W) where
            - N: number of groups (default = 1)
            - C: number of channels (=2 being real and imaginary parts)
            - H: height (x axis)
            - W: width (z axis)

        Output
        -----------------------------
        PyTorch tensor with dimensions (N,C,H,W)
        """
        return self.padding_zero_z(F.conv2d(x, self.gradient_e_z, padding=0, groups=2))

    def d_h_z(self, x):
        """
        Returns the first order z derivative on the Yee grid for the magnetic field

        Input
        ------------------------------
        x : PyTorch tensor with dimensions (N,C,H,W) where
            - N: number of groups (default = 1)
            - C: number of channels (=2 being real and imaginary parts)
            - H: height (x axis)
            - W: width (z axis)

        Output
        -----------------------------
        PyTorch tensor with dimensions (N,C,H,W)
        """
        return self.padding_zero_z(F.conv2d(x, self.gradient_h_z, padding=0, groups=2))

    def dd_x(self, x):
        """
        Returns the second order x derivative applied first on the electric and then on the magnetic grid

        Input
        ------------------------------
        x : PyTorch tensor with dimensions (N,C,H,W) where
            - N: number of groups (default = 1)
            - C: number of channels (=2 being real and imaginary parts)
            - H: height (x axis)
            - W: width (z axis)

        Output
        -----------------------------
        PyTorch tensor with dimensions (N,C,H,W)
        """
        return self.d_h_x(self.d_e_x(x))

    def dd_x_pml(self, x):
        """
        Returns the second order x derivative applied first on the electric and then on the magnetic grid.
        More than dd_x() it implements also perfectly matched layers condition at the borders

        Input
        ------------------------------
        x : PyTorch tensor with dimensions (N,C,H,W,D) where
            - N: number of groups (default = 1)
            - C: number of channels (=2 being real and imaginary parts)
            - H: height (x axis)
            - W: width (z axis)

        Output
        -----------------------------
        PyTorch tensor with dimensions (N,C,H,W)
        """
        return self.rx_inverse * self.d_h_x(self.rx_inverse * self.d_e_x(x))

    def dd_z(self, x):
        """
        Returns the second order z derivative applied first on the electric and then on the magnetic grid

        Input
        ------------------------------
        x : PyTorch tensor with dimensions (N,C,H,W) where
            - N: number of groups (default = 1)
            - C: number of channels (=2 being real and imaginary parts)
            - H: height (x axis)
            - W: width (z axis)

        Output
        -----------------------------
        PyTorch tensor with dimensions (N,C,H,W)
        """
        return self.d_h_z(self.d_e_z(x))

    def dd_z_pml(self, x):
        """
        Returns the second order z derivative applied first on the electric and then on the magnetic grid
        More than dd_z() it implements also perfectly matched layers condition at the borders

        Input
        ------------------------------
        x : PyTorch tensor with dimensions (N,C,H,W) where
            - N: number of groups (default = 1)
            - C: number of channels (=2 being real and imaginary parts)
            - H: height (x axis)
            - W: width (z axis)

        Output
        -----------------------------
        PyTorch tensor with dimensions (N,C,H,W)
        """
        return self.rz_inverse * self.d_h_z(self.rz_inverse * self.d_e_z(x))

    def dd_zx(self, x):
        """
        Returns the derivative along x and z applied first on the electric (x) and then on the magnetic (z) grid

        Input
        ------------------------------
        x : PyTorch tensor with dimensions (N,C,H,W) where
            - N: number of groups (default = 1)
            - C: number of channels (=2 being real and imaginary parts)
            - H: height (x axis)
            - W: width (z axis)

        Output
        -----------------------------
        PyTorch tensor with dimensions (N,C,H,W)
        """
        return self.d_h_z(self.d_e_x(x))


    def dd_zx_pml(self, x):
        """
        Returns the derivative along x and z applied first on the electric (x) and then on the magnetic (z) grid
        More than dd_zx() it implements also perfectly matched layers condition at the borders

        Input
        ------------------------------
        x : PyTorch tensor with dimensions (N,C,H,W) where
            - N: number of groups (default = 1)
            - C: number of channels (=2 being real and imaginary parts)
            - H: height (x axis)
            - W: width (z axis)

        Output
        -----------------------------
        PyTorch tensor with dimensions (N,C,H,W)
        """
        return self.rz_inverse * self.d_h_z(self.rx_inverse * self.d_e_x(x))


    def dd_xz(self, x):
        """
        Returns the derivative along z and x applied first on the electric (z) and then on the magnetic (x) grid

        Input
        ------------------------------
        x : PyTorch tensor with dimensions (N,C,H,W) where
            - N: number of groups (default = 1)
            - C: number of channels (=2 being real and imaginary parts)
            - H: height (x axis)
            - W: width (z axis)

        Output
        -----------------------------
        PyTorch tensor with dimensions (N,C,H,W)
        """
        return self.d_h_x(self.d_e_z(x))


    def dd_xz_pml(self, x):
        """
        Returns the derivative along z and x applied first on the electric (z) and then on the magnetic (x) grid
        More than dd_xz() it implements also perfectly matched layers condition at the borders

        Input
        ------------------------------
        x : PyTorch tensor with dimensions (N,C,H,W) where
            - N: number of groups (default = 1)
            - C: number of channels (=2 being real and imaginary parts)
            - H: height (x axis)
            - W: width (z axis)

        Output
        -----------------------------
        PyTorch tensor with dimensions (N,C,H,W)
        """
        return self.rx_inverse * self.d_h_x(self.rz_inverse * self.d_e_z(x))

    def _Ey_to_Hx(self, ey):
        return self.d_e_z(ey)/(self.omega*constants.MU0) * 1j

    def _Ey_to_Hz(self, ey):
        return -self.d_e_z(ey)/(self.omega*constants.MU0) * 1j

    def _Ey_to_Sz(self, ey):
        return - 0.5 * ey * self._Ey_to_Hx(ey).conj()

class complex_tensor(torch.Tensor):
    """Complex tensor class inheritates from torch.Tensor. It handles complex variables in MaxwellNet
    as fields and pml tensors. They are 4 dimensional Pytorch Tensors with shape (Nb, Nc, Nx, Nz) where:
    - Nb: batch size
    - Nc = 2 being real(0) and imaginary(1) parts
    - Nx: height in pixels (x axis)
    - Nz: width in pixels (z axis)
     """
    @staticmethod
    def __new__(cls, tensor, *args, **kwargs):
        if (type(tensor) != torch.Tensor and type(tensor) != np.ndarray):
            raise TypeError('complex_field must be initialized with a PyTorch tensor or Numpy array')
        elif type(tensor) == torch.Tensor:
            if tensor.dim() != 4:
                raise ValueError('please provide a PyTorch tensor with dimension 4 (Nb, Nc, Nx, Nz)')
        elif type(tensor) == np.ndarray:
            if tensor.ndim != 2:
                raise ValueError('please provide a Numpy ndarray with dimension 2 (Nx, Nz)')
            else:
                tensor_tmp = torch.zeros((1,2,tensor.shape[0],tensor.shape[1]), dtype=torch.float32, requires_grad=False)
                tensor_tmp[0, 0, :, :] = torch.from_numpy(np.real(tensor))
                tensor_tmp[0, 1, :, :] = torch.from_numpy(np.imag(tensor))
                tensor = tensor_tmp
        return super().__new__(cls, tensor, *args, **kwargs)

    def __init__(self, *args, **kwargs):
        super(complex_tensor, self).__init__()
        #return torch.zeros(*args, **kwargs)

    def __mul__(self, other):
        if hasattr(other, "__len__"):
            return _complex_multiplication(self, other)
        else:
            return _complex_scalar_multiplication(self, other)

    def conj(self):
        """Returns the complex conjugate of the tensor"""
        return torch.cat((self[:, 0:1, :, :], -self[:, 1:2, :, :]), 1)

    def to_np(self):
        """convert the tensor to a numpy complex ndarray

        Returns
        -------
        complex ndarray of size (Nb, Nx, Nz)
        """
        return self[:,0,:,:].detach().cpu().numpy() + 1j*self[:,1,:,:].detach().cpu().numpy()

    def real(self):
        return torch.Tensor(self[:, 0:1, :, :])

if __name__ == '__main__':

    x = complex_tensor(torch.zeros((1, 2, 3, 4)))
    y = x*x

    print('Finished')