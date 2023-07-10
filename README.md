# MaxwellNet

A physics driven neural network to perform finite difference frequency domain electromagnetic simulations.

This repository contains the code to reproduce the results in the
paper ["Predicting nonlinear optical scattering with physics-driven neural networks"](https://arxiv.org/abs/2208.05793).
The original version of the code by Joowon Lim can be found [here](https://github.com/limjoowon/maxwellnet).

## Package structure

The main code to run MaxwellNet training can be found in the script `train_maxwellnet.py` and it can be run from command
line with the following synthax:

`
python train_maxwellnet.py --directory <DATA_DIRECTORY>
`

`<DATA_DIRECTORY>` should contain the train and validation sets in uncompressed `.npz` format called `train.npz`
and `valid.npz`. These files contain an array `n` with size `[N,1,Nx,Ny]` corresponding to the `N` refractive index
distributions to use during training/validation. The directory should also contain a JSON file
called `specs_maxwell.json` with the parameters of the network, simulation domain and training scheduler.

The script `dataset_generator.py` contains some functions to create different shape datasets, e.g. the lenses used in the paper.


### Colab interface

If you want to run the training in Google colab, just upload the code on your Drive 
account and create a notebook with the following code:

<pre><code>from google.colab import drive
drive.mount('/content/drive', force_remount=True)
%cd /content/drive/MyDrive/"MAXWELLNET_DIRECTORY"
%run train_maxwellnet.py --directory "DATA_DIRECTORY"
</pre></code>

## Main modifications from original version

Beyond adding the intensity dependent refractive index and the tunable U-Net, the present code
present some modifications with respect to the original version presented in ["MaxwellNet: 
Physics-driven deep neural 
network training 
based on Maxwell’s equations"](https://aip.scitation.org/doi/10.1063/5.0071616)

The main ones are reported here below:

  - Reformatting of some functions in `MaxwellNet.py` as loss(), set_gradient_kernels() and set_pml_tensors() 
  - Introduction of smoothing kernel for strong refractive index contrast
  - `n` now contains the refractive index distribution and not the scattering potential
  - The U-Net core returns the fast oscillating scattered field and not the field envelope
  - Introduction of the class `complex_tensor` in `MaxwellNet.py` to handle complex fields as PyTorch tensors
  - Implementation of periodic boundary conditions and localized source options
  - Implementation of some subroutines contained in `utils.py`