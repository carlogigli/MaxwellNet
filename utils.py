import torch
import torch.backends.cudnn as cudnn

import numpy as np
import random
import os, json
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import rcParams
import constants
rcParams['font.family'] = ['Calibri','sans-serif']
rcParams['font.weight'] = 'light'
rcParams['axes.labelweight'] = 'light'
rcParams['figure.titleweight'] = 'light'
rcParams['axes.titleweight'] = 'light'
rcParams['mathtext.fontset'] = 'stix'
rcParams.update({'font.size': 16})

os.environ['KMP_DUPLICATE_LIB_OK']='True'
#matplotlib.use('TkAgg')

def fix_seed(seed, is_cuda):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if is_cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        cudnn.benchmark = False
        cudnn.deterministic = True


def get_spec_with_default(specs, key, default):
    try:
        return specs[key]
    except KeyError:
        return default

def load_comsol(filename, inputs, specs):

    component = {'x': 5, 'y': 6, 'z': 4}
    try:
        data = np.loadtxt(os.path.join(inputs['directory'], filename), skiprows=9, dtype=complex)

    except:
        with open(os.path.join(inputs['directory'], filename), 'r') as file:
            filedata = file.read()
            # Replace the target string
        filedata = filedata.replace('i', 'j')
        with open(os.path.join(inputs['directory'], filename), 'w') as file:
            file.write(filedata)
        data = np.loadtxt(os.path.join(inputs['directory'], filename), skiprows=9, dtype=complex)

    Nx = specs["PhysicalSpecs"]["Nx"]
    Nz = specs["PhysicalSpecs"]["Nz"]

    Z = np.real(np.reshape(data[:, 0], (Nx, Nz))) * 1e6
    X = np.real(np.reshape(data[:, 1], (Nx, Nz))) * 1e6

    field_e = np.zeros((X.shape[0],X.shape[1],3), dtype=complex)
    field_h = np.zeros((X.shape[0], X.shape[1], 3), dtype=complex)

    for idx, comp in enumerate(component.values()):
        field_e[:,:,idx] = np.reshape(data[:, comp], (Nx, Nz))
        field_h[:,:,idx] = np.reshape(data[:, comp+3], (Nx, Nz))

    n = np.real(np.reshape(data[:, -1], (Nx, Nz)))

    return Z, X, field_e, field_h, n

def get_field_from_torch(field_torch):
    #field_np = field_torch.permute(2, 3, 1, 0)[:, :, :, :].squeeze().clone().detach().cpu()

    #amplitude = torch.sum(field_np[:, :, :].pow(2), 2).pow(1 / 2).numpy()
    #real = field_np[:, :, 0].numpy()
    #imag = field_np[:, :, 1].numpy()
    #return amplitude, real, imag
    return field_torch.to_np()[0, :, :]

def load_model(model_name, specs):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_dict = torch.load(model_name)
    model = MaxwellNet(**specs["NetworkSpecs"], **specs["PhysicalSpecs"]).to(device)
    model.load_state_dict(model_dict['state_dict'])

    return model, device

def to_tensorboard(field, losses, epoch, mode, symmetry, writer, train_valid):

    if mode == 'te':
        ey = field['Ey'].to_np()[0, : , :]
        polarization = ['y']
    elif mode == 'tm':
        ex = field['Ex'].to_np()[0, : , :]
        ez = field['Ez'].to_np()[0, : , :]
        polarization = ['x', 'z']

    if symmetry is True:
        if mode == 'te':
            ey = np.concatenate((np.flipud(ey),ey))
        elif mode == 'tm':
            ex = np.concatenate((np.flipud(ex), ex))
            ez = np.concatenate((-np.flipud(ez), ez))

    for idx in range(len(polarization)):
        if polarization[idx] == 'y':
            image = ey
        elif polarization[idx] == 'x':
            image = ex
        elif polarization[idx] == 'z':
            image = ez

        amplitude = np.sqrt(np.abs(image)**2)
        amplitude = amplitude - np.min(amplitude)
        amplitude = amplitude / np.max(amplitude)
        writer.add_image(train_valid + '/' + mode + '/amplitude_' + polarization[idx], amplitude, epoch, dataformats='HW')

        real = np.real(image)
        real = real - np.min(real)
        real = real / np.max(real)
        writer.add_image(train_valid + '/' + mode + '/real_' + polarization[idx], real, epoch, dataformats='HW')

        imaginary = np.imag(image)
        imaginary = imaginary - np.min(imaginary)
        imaginary = imaginary / np.max(imaginary)
        writer.add_image(train_valid + '/' + mode + '/imaginary_' + polarization[idx], imaginary, epoch, dataformats='HW')

    writer.add_scalar(train_valid + '/' + mode, losses, epoch)

def save_checkpoint(state, directory, filename):
    model_directory = os.path.join(directory, 'model')
    if os.path.exists(model_directory) == False:
        os.makedirs(model_directory)
    torch.save(state, os.path.join(model_directory, filename + '.pt'))


def predict_field(model,ri,S0,component,specs):

    model.eval()
    prediction = model(ri, specs['PhysicalSpecs']['n2'] * S0)

    # if component == 'x' or component == 'y':
    #     pred_field = model.complex_multiplication(prediction,
    #                                               model.fast[0:1, :, model.pad:-model.pad, model.pad:-model.pad])
    # else:
    #     pred_field = model.complex_multiplication(prediction[0:1, 2:4, :, :],
    #                                                   model.fast_z[0:1, :, model.pad:-model.pad, model.pad:-model.pad])
    pred_field = prediction
    loss = model.loss(prediction, ri, S0)
    print('Loss: {:.2e}'.format(torch.mean(loss.pow(2))))

    return pred_field, loss


def _complex_scalar_multiplication(a, b):

    c = torch.zeros((2)).to(a.device)
    c[0] = np.real(b)
    c[1] = np.imag(b)

    r_p = torch.mul(a[:, 0:1, :, :], c[0:1]) - torch.mul(a[:, 1:2, :, :], c[1:2])
    i_p = torch.mul(a[:, 0:1, :, :], c[1:2]) + torch.mul(a[:, 1:2, :, :], c[0:1])
    return torch.cat((r_p, i_p), 1)

def _complex_multiplication(a, b):
    """
    Returns the product of two complex tensors

    Input
    ------------------------------
    a,b : PyTorch tensors with dimensions (N,C,H,W,D) where C(number of channels)=2 being real and imaginary parts

    Output
    -----------------------------
    PyTorch tensor with dimensions (N,C,H,W,D) where C(number of channels)=2 being real and imaginary parts
    """
    r_p = torch.mul(a[:, 0:1, :, :], b[:, 0:1, :, :]) - torch.mul(a[:, 1:2, :, :], b[:, 1:2, :, :])
    i_p = torch.mul(a[:, 0:1, :, :], b[:, 1:2, :, :]) + torch.mul(a[:, 1:2, :, :], b[:, 0:1, :, :])
    return torch.cat((r_p, i_p), 1)


def plot_XZ(ax, image, specs, **options):

    physical_specs = specs['PhysicalSpecs']
    Nx = physical_specs['Nx']
    Nz = physical_specs['Nz']
    dpl = physical_specs['dpl']
    wavelength = physical_specs['wavelength']
    nb = physical_specs['nb']
    n2 = physical_specs['n2']
    symmetry_x = physical_specs['symmetry_x']
    mode = physical_specs['mode']
    pml_thickness = physical_specs['pml_thickness']

    delta = wavelength / dpl / nb  # Discretization step

    x = np.linspace(-Nx // 2, Nx // 2 - 1, Nx) * delta
    z = np.linspace(-Nz // 2, Nz // 2 - 1, Nz) * delta
    zz, xx = np.meshgrid(z, x)

    ax.set_xlabel(r'Z axis $(\mu m)$')
    ax.set_ylabel(r'X axis $(\mu m)$')

    if 'cmap' in options:
        cmap = options['cmap']
    else:
        cmap = 'jet'

    if symmetry_x:
        im = ax.pcolormesh(zz, xx, np.vstack((np.flipud(image),image)), cmap=cmap, rasterized=True)
    else:
        im = ax.pcolormesh(zz, xx, image, cmap=cmap, rasterized=True)
    divider = make_axes_locatable(ax)
    ax.set_aspect('equal')
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(im, cax=cax)
    plt.box()

    if 'vmin' and 'vmax' in options:
        im.set_clim(options['vmin'], options['vmax'])
    if 'title' in options:
        ax.set_title(options['title'])
    if 'hide_pml' in options:
        if options['hide_pml']:
            ax.set_xlim([z.min() + pml_thickness * delta, z.max() - pml_thickness * delta])
            ax.set_ylim([x.min() + pml_thickness * delta, x.max() - pml_thickness * delta])

    if 'shape' in options:
        import cv2
        data = options['shape'] - physical_specs['nb']
        data = data / data.max() * 255
        shape = np.uint8(data)
        contours, _ = cv2.findContours(shape, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            ax.fill(zz[c[:, 0, 1], c[:, 0, 0]], xx[c[:, 0, 1], c[:, 0, 0]], edgecolor='k', fill=False, linewidth=1.2)

    return im

if __name__ == '__main__':

    inputs = {
        'directory': './data/diffuser/test_2_tm',
        'dataset': 'valid',
        'idx': 0,
        'S0': 8e17,
        'component': 'y',
        'model': 'latest',
        'hide_pml': True,
        'cmax': 2
    }

    specs_filename = os.path.join(inputs['directory'], 'specs_maxwell.json')
    specs = json.load(open(specs_filename))

    model_directory = os.path.join(inputs['directory'], 'model')
    filename = inputs['model']
    model, device = load_model(os.path.join(model_directory, filename + '.pt'), specs)

    dataset = inputs['dataset']
    test_set = np.load(os.path.join(inputs['directory'], dataset + ".npz"))['n']

    idx = inputs['idx']

    S0 = inputs['S0']
    S0_torch = torch.from_numpy(np.array([S0])).to(device)
    ref_index = torch.from_numpy(test_set[idx:idx + 1, 0:1, :, :]).to(device)

    pred_field, loss = predict_field(model, ref_index, S0_torch, inputs['component'], specs)

    amplitude, real, imag = get_field_from_torch(pred_field)

    fig, ax = plt.subplots(1, 3, figsize = (15, 8))

    plot_XZ(ax[0],amplitude,specs,title='amplitude', vmin = 0, vmax = inputs['cmax'], cmap = 'Greys')
    plot_XZ(ax[1], real, specs)
    plot_XZ(ax[2], imag, specs)

    plt.tight_layout()
    plt.show()