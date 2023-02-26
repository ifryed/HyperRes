"""
Training script for HyperRes: Hypernetwork-Based Adaptive Image Restoration
Paper: https://arxiv.org/abs/2206.05970
By Shai Aharon ifryed@gmail.com
"""
from __future__ import print_function

import argparse
import itertools

import shutil
import time
import os
from itertools import chain

import functools
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

import numpy as np
from models.Unet import RDUNet, init_weights
from models.HyperRes import HyperRes
from utils import AverageMeter, Logger
from utils.DataUtils import CommonTools
from utils.DataUtils.CommonTools import calculate_psnr, saveImage, weights_init_kaiming, postProcessForStats
from utils.DataUtils.TrainLoader import NoisyDataset
import cv2

parser = argparse.ArgumentParser(description='PyTorch Multiple Objectives Network Training')
parser.add_argument('--data', metavar='DIR', help='path to dataset')
parser.add_argument('--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 0)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='Set manual epoch number (for resuming)')
parser.add_argument('--batch-size', default=16, type=int,
                    metavar='N',
                    help='Mini-batch size (default: 16)')
parser.add_argument('--lr', default=1e-4, type=float, metavar='LR', help='Initial learning rate',
                    dest='lr')
parser.add_argument('--weight-decay', default=0, type=float,
                    metavar='W', help='Weight decay (default: 0)',
                    dest='weight_decay')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='Path to the latest checkpoint (default: none)')
parser.add_argument('--seed', default=0, type=int, help='seed for initializing training. ')
parser.add_argument('--schedule', type=int, nargs='+', default=[500000], help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--norm_f', type=float, default=255, help='The normalization factor for the distortion levels.')
parser.add_argument('--checkpoint', default='pre_trained', type=str, metavar='PATH',
                    help='Path to save checkpoint (default: pre_trained)')
parser.add_argument('--lvls', type=int, nargs='+', default=[15], help='A list of corruptions levels to train on')
parser.add_argument('--meta_blocks', type=int, default=16, help='Number of Meta Blocks (default: 16)')
parser.add_argument('--steps', type=int, default=1000000, help='Number of total steps')
parser.add_argument('--valid', type=str, default='valid',
                    help='Which dataset to use for evaluation, Validation or Test')
parser.add_argument('--device', type=str, default='cpu', help='Device to run on,[cpu,cuda,cuda:0..]')
parser.add_argument('--val_iter', type=int, nargs='+', default=[1, 10, 10, 50],
                    help='Run validation every X iterations, last interval is applied until the end of the training')
parser.add_argument('--no_bias', dest='bias', default=True, action='store_false',
                    help='Add Bias in the ResBlocks')
parser.add_argument('--parallel','-p', dest='parallel', default=False, action='store_true',
                    help='Check to run on parallel GPUs')
parser.add_argument('-y', '-Y', '--gray', dest='y_channel', default=False, action='store_true',
                    help='Train on Grayscale only')
parser.add_argument('--data_type', type=str, default='n', choices=['n', 'sr', 'j'],
                    help='Defines the task data, de(n)oise, super-resolution(sr), de(j)peg.')

def main():
    global args

    # Args handling
    args = parser.parse_args()

    # Create model
    CommonTools.set_random_seed(args.seed)
    print("=> Assembling model 'HyperRes Network'")
    model = HyperRes(meta_blocks=args.meta_blocks, level=args.lvls, device=args.device,
                     gray=args.y_channel, norm_factor=args.norm_f)
    # model = RDUNet(levels=args.lvls)

    # Weight Normalization
    model.apply(functools.partial(weights_init_kaiming, scale=0.1))
    # model.apply(functools.partial(init_weights()))
    model.to(args.device)
    if args.parallel:
        model = torch.nn.DataParallel(model).to(args.device)

    # Define loss function (criterion) and optimizer
    criterion = nn.L1Loss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)

    # Optionally resume from a checkpoint
    title = 'UnetHyperRes'
    if not os.path.isdir(args.checkpoint):
        os.makedirs(args.checkpoint, exist_ok=True)

    best_prec1 = 0
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            args.checkpoint = os.path.dirname(args.resume)
            logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title, resume=True)
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
        logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])

    torch.backends.cudnn.benckmark = True

    # Data loading code
    crop_size = 96
    args.crop_size = crop_size
    train_loader = NoisyDataset(
        os.path.join(args.data, 'train'),
        cor_lvls=args.lvls,
        crop_size=crop_size,
        phase='train', 
        lr_prefix=args.data_type,
        interp=False
        )
    data_set_len = len(train_loader)
    train_loader = torch.utils.data.DataLoader(
        train_loader,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)

    val_loader = NoisyDataset(
        os.path.join(args.data, args.valid),
        cor_lvls=args.lvls,
        phase='test',
        lr_prefix=args.data_type,
        interp=False
        )
    val_loader = torch.utils.data.DataLoader(
        val_loader,
        batch_size=1, shuffle=False,
        num_workers=1, pin_memory=True)
    val_loader_len = len(val_loader)

    tot_epochs = int(np.ceil(args.steps / data_set_len)) * 16 + 5
    args.epochs = tot_epochs
    val_idx = 0

    global curr_step
    curr_step = 0

    # Training loop
    for epoch in range(args.start_epoch, tot_epochs):
        print('\nEpoch: [Epoch: {:d}/{:d}, Step: {:d}/{:d}]'.format(epoch + 1, tot_epochs, curr_step, args.steps))
        print('LR: {}'.format(optimizer.param_groups[0]['lr']))

        # Train for one epoch
        train_loss, train_acc = train(train_loader, model, criterion, optimizer)
        print("=============")

        # Evaluate on validation set
        is_best = False

        if epoch % 2 == 0:
            val_idx += 1
            last_val = epoch
            val_loss, prec1 = evaluate(val_loader, val_loader_len, model, criterion)
            print("Validation PSNR:")
            [print("\t{}: {:.3f}".format(lvl, psnr)) for lvl, psnr in zip(args.lvls, prec1)]

            lr = optimizer.param_groups[0]['lr']

            # append logger file
            logger.append([lr, train_loss, val_loss, train_acc[0], prec1[0]])

            is_best = sum(prec1) / len(prec1) > best_prec1
            best_prec1 = max(sum(prec1) / len(prec1), best_prec1)

            if curr_step > args.steps:
                break

        save_checkpoint({
            'epoch': epoch + 1,
            'arch': 'Unet',
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
            'sigmas': args.lvls,
        }, is_best, checkpoint=args.checkpoint, epoch=epoch)

    # Run Test
    test_loader = NoisyDataset(
        os.path.join(args.data, 'test'),
        cor_lvls=args.lvls,
        phase='test', lr_prefix=args.data_type)
    test_loader = torch.utils.data.DataLoader(
        test_loader,
        batch_size=1, shuffle=False,
        num_workers=1, pin_memory=True)
    test_loader_len = len(test_loader)

    loss_avg, psnr_avg = evaluate(test_loader, test_loader_len, model, criterion)
    print("Test PSNR")
    [print("\t{}: {:.3f}".format(lvl, psnr)) for lvl, psnr in zip(args.lvls, psnr_avg)]

    logger.close()

    print('Best accuracy:')
    print(best_prec1)


def train(train_loader, model, criterion, optimizer):
    global curr_step
    losses = AverageMeter()
    sig_loss = [AverageMeter() for _ in range(len(args.lvls))]

    # switch to train mode
    model.train()

    sig_frac = np.array([1.0 / len(args.lvls) for _ in args.lvls])
    st = time.time()
    for i, data in enumerate(train_loader):
        updateLR(curr_step, optimizer)

        target = data['target'].to(args.device, non_blocking=True)
        target = [target for _ in range(len(args.lvls))]
        images = [x.to(args.device) for x in data['image']]

        # Compute output
        optimizer.zero_grad()
        output = model(images)
        loss = 0
        for j in range(len(args.lvls)):
            loss += criterion(output[j], target[j]).mul(sig_frac[j])

        # Compute gradient and do SGD step
        loss.backward()
        optimizer.step()

        log_stat = {"Loss": loss.item(), 'LR': optimizer.param_groups[0]['lr']}
        curr_step += 1 #train_loader.batch_size

        losses.update(loss.item())
        if i % 10 == 0:
            print("\rStep: {}|\tLoss: {:.4f}".format(curr_step, losses.avg), end='')
        if curr_step > args.steps:
            break

    print("\nEpoch Time: {:.3f}".format(time.time() - st))
    return losses.avg, [t.avg for t in sig_loss]


def updateLR(curr_step, optimizer):
    lr = args.lr * 0.1 ** sum([curr_step > x for x in args.schedule])
    optimizer.param_groups[0]['lr'] = lr


def evaluate(val_loader, val_loader_len, model, criterion,val_phase='Test'):
    print("Running Evaluation...")

    losses = AverageMeter()
    lvl_loss = [AverageMeter() for _ in range(len(args.lvls))]

    # Switch to evaluate mode
    model.eval()

    smp_plot_n = len(args.lvls)
    loss = 0
    wand_img = []
    random_image = np.random.randint(val_loader_len)
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            target = data['target'].to(args.device, non_blocking=True)
            images = [x.to(args.device) for x in data['image']]

            # Compute output
            output = model(images)
            for j in range(len(args.lvls)):
                loss += criterion(output[j], target)

                if i == random_image:
                    im1 = saveImage(
                            output[j].detach().cpu().numpy()[0],
                            images[j].detach().cpu().numpy()[0],
                            j)

                # Measure PSNR
                for out_idx, out in enumerate(output[j]):
                    imgs = postProcessForStats([target[out_idx], out])
                    trg, out = imgs
                    psnr = calculate_psnr(out, trg)
                    lvl_loss[j].update(psnr)
            losses.update(loss.item())

    # Switch back to train mode
    model.train()

    return losses.avg, [t.avg for t in lvl_loss]


def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth', epoch=None):
    filepath = os.path.join(checkpoint, filename)
    
    torch.save(state, filepath)
    shutil.copyfile(filepath, os.path.join(checkpoint, 'latest.pth'))
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth'))


if __name__ == '__main__':
    main()
