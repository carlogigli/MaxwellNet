import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os, json
import cv2
matplotlib.use('TkAgg')
from matplotlib.patches import Rectangle
from utils import *
import constants
from tqdm import tqdm
from autograd.scipy.signal import convolve as conv
from skimage.draw import circle

def randrange(x1,x2):
    delta = x2-x1
    return x1 + delta * np.random.rand()

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def get_circle_idx(x,y,x0,y0,r):
    '''
    It returns the indexes of a circle centered in (x0,y0)
    '''
    return ((x-x0)**2+(y-y0)**2)<(r**2)

def operator_lowpass(rho, fmax):
    """
    Low pass filter to cut spatial spectrum at fmax
    """
    Ny, Nx = rho.shape
    Kx, Ky = np.meshgrid(np.linspace(-1,1,Nx)*np.pi/Nx,np.linspace(-1,1,Ny)*np.pi/Ny)
    circular_aperture = (Kx ** 2 + Ky ** 2) < fmax ** 2
    rho_ft = np.fft.fftshift(np.fft.fft2(rho))
    return np.abs((np.fft.ifft2(rho_ft * circular_aperture)))

def sag_profile(r2,roc,kcon):
    return r2/(roc*(1+np.sqrt(1-((1+kcon)*r2)/roc**2)))

def randprofile(Nx,Nz,filter,dp):

    profile = np.random.rand(Nx)
    profile_ft = np.fft.fftshift(np.fft.fft(profile))

    profile = np.abs(np.fft.ifftshift(np.fft.ifft(profile_ft * filter)))
    #profile = profile - profile.min()
    profile = profile - profile.mean()
    profile = profile / (profile.max()-profile.min()) * dp
    return np.tile(profile,(Nz,1)).T

def create_lens(xx,zz,profile_left,profile_right,thickness,rpupil, beta=100):
    return sigmoid(beta*(-zz+profile_right+thickness/2))*sigmoid(beta*(zz-profile_left+thickness/2))*sigmoid(beta*(xx+rpupil))*sigmoid(beta*(-xx+rpupil))

def create_layer(zz,profile_left,profile_right,thickness):
    return (zz-thickness/2 < profile_right)*(zz+thickness/2 > profile_left)

def diffuser_dset(geometry, params, Nsamples):
    Smin = 2e17  # [W/m2]
    Smax = 10e17  # [W/m2]

    shapes = []
    power = []

    dtmin = 2 * physical_specs['wavelength'] / params['nm']
    dtmax = 2 * physical_specs['wavelength'] / params['nm']
    dp = physical_specs['wavelength'] / params['nm'] / 2
    z0 = 0

    kx = np.linspace(-Nx // 2, Nx // 2 - 1, Nx) * 2 * np.pi / Lx
    kmax = 8 * 2 * np.pi / Lx
    square_aperture = (kx ** 2) < kmax ** 2

    l = 4
    borders = (np.abs(geometry['X']) < l / 2) * (np.abs(geometry['Z']) < l / 2)

    for i in tqdm(range(Nsamples)):
        shape = np.zeros((Nz, Nx))
        t1 = randrange(dtmin, dtmax)
        shape = shape + create_layer(zz - z0, -randprofile(Nx, Nz, square_aperture, dp),
                              randprofile(Nx, Nz, square_aperture, dp), t1)
        shape = (shape > 0.5) * borders
        shapes.append(shape)
        power.append(randrange(Pmin, Pmax))

    n = np.asarray(shapes) * (params['nm'] - params['nb']) + params['nb']
    return n, power

def lenses_dset(geometry, params, Nsamples):

    Smin = 5e2  # [W/m2]
    Smax = 5e5  # [W/m2]

    shapes = []
    power = []

    roc_min = geometry['Lx']/2
    roc_max = geometry['Lx']*2
    t_min = params['wl'] / params['nb'] * 3
    t_max = params['wl'] / params['nb'] * 4.5
    h = params['wl']/params['nb']*10

    for i in tqdm(range(Nsamples)):

        shape = np.zeros((geometry['Nx'], geometry['Nz']))
        x2 = geometry['X'] ** 2
        pr_left = sag_profile(x2, randrange(roc_min,4*roc_max), randrange(-4,0))
        pr_right = sag_profile(x2, -randrange(roc_min,roc_max), randrange(-4,0))
        shapes.append(create_lens(geometry['X'], geometry['Z'], pr_left, pr_right, randrange(t_min,t_max), h/2, beta=500))
        power.append(randrange(Smin, Smax))

    n = np.asarray(shapes) * (params['nm'] - params['nb']) + params['nb']

    return n, power


def save_dset(directory, n, power, src = None, chi3 = None):

    Nsamples = len(n)
    print('Library size: {}'.format(Nsamples))
    n = np.asarray(n, dtype=np.float32)[:, np.newaxis, :, :]
    train_valid_ratio = .8

    if Nsamples == 1:
        train = n
        train_p = np.asarray(power)
        if src is None:
            np.savez(os.path.join(directory, 'train.npz'), n=train, S0=train_p)
        elif chi3 is None:
            np.savez(os.path.join(directory, 'train.npz'), n=train, S0=train_p, src=src)
        else:
            np.savez(os.path.join(directory, 'train.npz'), n=train, S0=train_p, src=src, chi3=chi3)

    else:
        train_idx = np.random.permutation(int(Nsamples * train_valid_ratio))
        train = n[:int(n.shape[0] * train_valid_ratio), :, :, :][train_idx]
        train_p = np.asarray(power[:int(n.shape[0] * train_valid_ratio)])[train_idx]

        valid_idx = np.random.permutation(int(Nsamples - len(train)))
        valid = n[int(n.shape[0] * train_valid_ratio):, :, :, :][valid_idx]
        valid_p = np.asarray(power[int(n.shape[0] * train_valid_ratio):])[valid_idx]

        if src is None:
            np.savez(os.path.join(directory, 'train.npz'), n=train, S0=train_p)
            np.savez(os.path.join(directory, 'valid.npz'), n=valid, S0=valid_p)
        elif chi3 is None:
            np.savez(os.path.join(directory, 'train.npz'), n=train, S0=train_p, src=src)
            np.savez(os.path.join(directory, 'valid.npz'), n=valid, S0=valid_p, src=src)
        else:
            np.savez(os.path.join(directory, 'train.npz'), n=train, S0=train_p, src=src, chi3=chi3)
            np.savez(os.path.join(directory, 'valid.npz'), n=valid, S0=valid_p, src=src, chi3=chi3)

    fig, ax = plt.subplots()
    if src is None:
        im = ax.pcolormesh(zz, xx, n[0, 0, :, :])
    else:
        im = ax.pcolormesh(zz, xx, n[0, 0, :, :]*(src==0))
    ax.add_patch(Rectangle((-Lz / 2, -Lx / 2), Lz - dz, pml_thickness * dx, fill=False, hatch='/'))
    ax.add_patch(
        Rectangle((-Lz / 2, Lx / 2 - pml_thickness * dx - dx), Lz - dz, pml_thickness * dx, fill=False, hatch='/'))
    ax.add_patch(
        Rectangle((-Lz / 2, -Lx / 2 + pml_thickness * dx), pml_thickness * dz, Lx - 2 * pml_thickness * dx - dx,
                  fill=False, hatch='/'))
    ax.add_patch(Rectangle((Lz / 2 - pml_thickness * dz - dz, -Lx / 2 + pml_thickness * dx), pml_thickness * dz,
                           Lx - 2 * pml_thickness * dx - dx, fill=False, hatch='/'))
    plt.colorbar(im)
    plt.show()

if __name__ == '__main__':

    directory = './data/microlenses'

    specs_filename = os.path.join(directory,'specs_maxwell.json')
    specs = json.load(open(specs_filename))
    physical_specs = specs["PhysicalSpecs"]

    dx = physical_specs['wavelength'] / physical_specs['dpl'] / physical_specs['nb'] # x-resolution
    dz = dx  # y-resolution
    pml_thickness = physical_specs['pml_thickness']
    if physical_specs['symmetry_x']:
        Nx = physical_specs['Nx']  # x-grid size
        #x = np.linspace(-Nx // 2, 0, Nx//2) * dx
        x = np.linspace(0, Nx//2, Nx // 2) * dx
    else:
        Nx = physical_specs['Nx']  # x-grid size
        x = np.linspace(-Nx // 2, Nx // 2 - 1, Nx) * dx

    Nz = physical_specs['Nz']  # z-grid size
    Lx = Nx * dx  # x-axis length
    Lz = Nz * dz  # z-axis length
    z = np.linspace(-Nz // 2, Nz // 2 - 1, Nz) * dz
    zz, xx = np.meshgrid(z,x)

    geometry = {'Z': zz, 'X': xx, 'Nx': Nx, 'Nz': Nz, 'Lz': Lz, 'Lx': Lx, 'dz': dz, 'dx': dx, 'Npml': physical_specs['pml_thickness']}
    params = {'wl': physical_specs['wavelength'], 'nm': 1.52, 'nb': physical_specs['nb'], 'symmetry_x': physical_specs['symmetry_x']}
    Nsamples = 4000
    n, power = lenses_dset(geometry, params, Nsamples)
    #n, power, src, chi3 = pcw_dset(geometry, params, Nsamples)
    save_dset(directory, n, power)
