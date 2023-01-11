# Copyright (c) 2021 Joowon Lim - Carlo Gigli

import torch
from Datasets import RI2D_Dataset, WG2D_Dataset
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from MaxwellNet import MaxwellNet

import numpy as np
import logging
import argparse
import os
import time
from datetime import timedelta
import constants
import json
from utils import fix_seed, get_spec_with_default, get_field_from_torch, to_tensorboard, save_checkpoint, load_model

def main(directory, is_server, load_latest):
    """
    Main routine to train MaxwellNet

    :param directory: path where the datasets and spec.json file are saved
    :param is_server: boolean variable, set it true if training on a server
    :param load_latest: boolean variable, if true it starts back the training from a pretrained model in directory folder
    :return:
    """

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    logging.info("training " + directory)

    specs_filename = os.path.join(directory, 'specs_maxwell.json')

    if not os.path.isfile(specs_filename):
        raise Exception(
            'The experiment directory does not include specifications file "specs_maxwell.json"'
        )

    specs = json.load(open(specs_filename))
    seed_number = get_spec_with_default(specs, "Seed", None)
    if seed_number != None:
        fix_seed(seed_number, torch.cuda.is_available())

    if is_server:
        rank = int(os.environ['SLURM_PROCID'])
    else:
        import matplotlib
        import matplotlib.pyplot as plt
        matplotlib.use('TkAgg')
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
        logging.getLogger('matplotlib').setLevel(logging.ERROR)
        logging.getLogger('PIL').setLevel(logging.WARNING)
        logging.getLogger('h5py').setLevel(logging.WARNING)
        rank = 0

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logging.info("Experiment description: \n" + ' '.join([str(elem) for elem in specs["Description"]]))
    logging.info("Training with " + str(device))

    batch_size = get_spec_with_default(specs, "BatchSize", 1)
    epochs = get_spec_with_default(specs, "Epochs", 1)
    snapshot_freq = specs["SnapshotFrequency"]
    physical_specs = specs["PhysicalSpecs"]
    symmetry_x = physical_specs['symmetry_x']
    mode = physical_specs['mode']
    high_order = physical_specs['high_order']

    train_dataset = RI2D_Dataset(directory, 'train')
    n0 = train_dataset.__getitem__(0)[0].min() # Background refractive index
    PhySpecs = specs["PhysicalSpecs"]
    PhySpecs['nb'] = n0
    PhySpecs['n2'] = PhySpecs['n2']

    if is_server:
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, sampler=None, num_workers=4)
    else:
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, sampler=None)
    logging.info("Train Dataset length: {}".format(len(train_dataset)))

    loss_train = torch.zeros((int(epochs),), dtype=torch.float32, requires_grad=False)

    if len(train_dataset) > 1:
        perform_valid = True
    else:
        perform_valid = False

    if perform_valid == True:
        valid_dataset = RI2D_Dataset(directory, 'valid')
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size,
                                                   shuffle=True, pin_memory=True, sampler=None)
        logging.info("Valid Dataset length: {}".format(len(valid_dataset)))
        loss_valid = torch.zeros((int(epochs),), dtype=torch.float32, requires_grad=False)

    model = MaxwellNet(**specs["NetworkSpecs"], **PhySpecs)

    if 'loc_source' in physical_specs:
        if physical_specs['loc_source']:
            src = train_dataset.loc_src
            model.insert_localized_source(src)
    #model.set_inhomogeneous_chi3(train_dataset.chi3)

    if os.path.exists(os.path.join(directory, 'model\latest.pt')) and load_latest==True:
        model_dict = torch.load(os.path.join(directory, 'model\latest.pt'))
        model.load_state_dict(model_dict['state_dict'])

    if torch.cuda.device_count() > 1:
        logging.info("Multiple GPUs: " + str(torch.cuda.device_count()))
    model = torch.nn.DataParallel(model)
    model = model.to(device)

    logging.info("Number of network parameters: {}".format(sum(p.data.nelement() for p in model.parameters())))
    logging.debug(specs["NetworkSpecs"])
    logging.debug(specs["PhysicalSpecs"])

    optimizer = torch.optim.Adam(model.parameters(), lr=get_spec_with_default(specs, "LearningRate", 0.0001), weight_decay=0)
    scheduler = StepLR(optimizer, step_size=get_spec_with_default(specs, "LearningRateDecayStep", 10000), gamma=get_spec_with_default(specs, "LearningRateDecay", 1.0))

    checkpoints = list(range(snapshot_freq, epochs + 1, snapshot_freq))

    filename = 'maxwellnet_' + mode + '_' + high_order
    writer_freq = get_spec_with_default(specs, "TensorboardFrequency", None)

    tic = time.time()
    logging.info("Training start")
    if not is_server:
        fig, axs = plt.subplots(2, 2)
        plt.tight_layout()
    for epoch in range(1, epochs + 1):

        e_tot_train, train_loss, diff = train(train_loader, model, optimizer, device, specs["PhysicalSpecs"])
        loss_train[epoch - 1] = train_loss

        if not is_server and epoch % 50 == 0:
            e_tot = (e_tot_train['Ex']+e_tot_train['Ey'])[0:1, 0:2, :, :]

            axs[0,0].clear()
            field = (e_tot_train['Ex']+e_tot_train['Ey']).to_np()[0,:,:]
            cmax = np.abs(field).max()
            axs[0,0].imshow(np.real(field), cmap='turbo', vmin=-cmax, vmax=cmax)
            axs[0,0].set_title(r'$Re[E]$')
            axs[0,1].imshow(np.abs(field), cmap='turbo', vmin=0, vmax=cmax)
            axs[0,1].set_title(r'$|E|$ Epoch: {:d}'.format(epoch))
            axs[1,0].imshow(np.imag(field), cmap='turbo', vmin=-cmax, vmax=cmax)
            axs[1,0].set_title(r'$Im[E]$')
            axs[1, 1].imshow(np.abs(diff.to_np()[0,:,:]), cmap='turbo')
            axs[1, 1].set_title(r'Loss')
            fig.savefig(os.path.join(directory,'field_train.png'))

        if perform_valid:
            e_tot_valid, valid_loss = valid(valid_loader, model, device, specs["PhysicalSpecs"])
            loss_valid[epoch - 1] = valid_loss

        if epoch % writer_freq == 0:
            logging.info("Epoch: {:d}/{:d} | Elapsed: {} | Loss: {:.2e} | lr: {:.2e}".format(epoch, epochs, str(timedelta(seconds=time.time() - tic)), loss_train[epoch - 1].item(), scheduler.get_last_lr()[0]))
            with open(os.path.join(directory, 'train_loss.txt'), "ab") as f:
                np.savetxt(f, np.array([epoch, loss_train[epoch - 1]], ndmin=2), fmt='%d %.3e')

            if perform_valid:
                with open(os.path.join(directory, 'valid_loss.txt'), "ab") as f:
                    np.savetxt(f, np.array([epoch, loss_valid[epoch - 1]], ndmin=2), fmt='%d %.3e') \


        if epoch in checkpoints:
            if rank == 0:
                save_checkpoint({
                    'epoch': epoch,
                    'state_dict': model.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'loss_train': loss_train,
                    'scheduler': scheduler.state_dict(),
                }, directory, str(epoch) + '_' + mode + '_' + high_order)

        if epoch % 200 == 0:
            if rank == 0:
                save_checkpoint({
                    'epoch': epoch,
                    'state_dict': model.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'loss_train': loss_train,
                    'scheduler': scheduler.state_dict(),
                }, directory, 'latest')

        scheduler.step()

def train(train_loader, model, optimizer, device, PhySpecs):
    model.train()
    with torch.set_grad_enabled(True):
        count = 0
        train_loss = 0
        for data in train_loader:
            ref_index = data[0].to(device)
            S0 = data[1].to(device)
            total = model(ref_index, S0*PhySpecs['n2'])  # [N, 2, H, W]
            diff = model.module.loss(total, ref_index, S0)
            l2 = diff.pow(2)
            loss = torch.mean(l2)
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1e-3)
            optimizer.step()

            train_loss += loss.item() * diff.size(0)
            count += diff.size(0)

        train_loss = train_loss / count

    return total, train_loss, diff

def valid(valid_loader, model, device, PhySpecs):
    model.eval()
    with torch.set_grad_enabled(False):
        count = 0
        valid_loss = 0
        for data in valid_loader:
            ref_index = data[0].to(device)
            S0 = data[1].to(device)
            total = model(ref_index, S0*PhySpecs['n2'])
            diff = model.module.loss(total, ref_index, S0)
            l2 = diff.pow(2)
            loss = torch.mean(l2)

            valid_loss += loss.item() * diff.size(0)
            count += diff.size(0)

        valid_loss = valid_loss / count

    return total, valid_loss

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description="Train a MaxwellNet")
    arg_parser.add_argument(
        "--directory",
        "-d",
        required=False,

        default='./data/lc_lenses/tm',
        help="This directory should include "
             + "all the training and network parameters in 'specs.json', and logging will be "
             + "done in this directory as well.",
    )
    arg_parser.add_argument('--is_server', action='store_true', default=False,
                        help='true if it runs on server')
    arg_parser.add_argument('--load_latest', action='store_true', default=False,
                            help='if true and a pretrained model exists in the directory, it will load it before starting the training')

    args = arg_parser.parse_args()
    main(args.directory, args.is_server, args.load_latest)

