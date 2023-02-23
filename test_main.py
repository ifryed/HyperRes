"""
Training script for Multiple Objectives Network (MoNet)
"""
from __future__ import print_function

import argparse
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import os

import torch.optim
import torch.utils.data
import torch.utils.data.distributed

from models import HyperRes
from utils import AverageMeter
from utils.DataUtils import CommonTools
from utils.DataUtils.CommonTools import calculate_psnr, postProcessForStats, saveImage
from utils.DataUtils.TrainLoader import NoisyDataset
import numpy as np

from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


parser = argparse.ArgumentParser(description='PyTorch ImageNet Testing')
parser.add_argument('--data', metavar='DIR', help='path to dataset')
parser.add_argument('--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--weights', required=True, type=str, metavar='PATH',
                    help='Path to latest checkpoint')
parser.add_argument('--meta_blocks', type=int, default=16, help='Number of Meta Blocks (default: 16)')
parser.add_argument('--valid', type=str, default='valid', help='Evaluation folder')
parser.add_argument('--device', type=str, default='cpu', help='Device to run on,[cpu,cuda..]')
parser.add_argument('--lvls', type=int, nargs='+', default=[15], help='A list of corruptions levels to evaluate')
parser.add_argument('-y', '-Y', '--gray', dest='y_channel', default=False, action='store_true',
                    help='Test on Grayscale')
parser.add_argument('--sample', dest='save_sample', default=False, action='store_true', help='Save sample images')
parser.add_argument('--data_type', type=str, default='n', choices=['n', 'sr', 'j'],
                    help='Defines the task data, de(n)oise, super-resolution(sr), de(j)peg.')
parser.add_argument('--norm_f', type=float, default=255, help='The normalization factor for the distortion levels.')


def getBounds(lvls_arr: list, n_lvl: float):
    """
    A helper function to calculate the BN interpolation
    @param lvls_arr: The corruption levels list
    @param n_lvl: The current level to interpolate
    @return: Returns the post and previous corruption levels
    """
    lower = lvls_arr[0]
    upper = lvls_arr[1]
    for i, v in enumerate(lvls_arr[:-1]):
        if n_lvl <= v:
            break
        lower = v
        upper = lvls_arr[i + 1]

    return lower, upper


def bn_interpolate(state_dict: dict, orig_lvls: list, new_lvls: list):
    """
    Interpolates the Batch Normalization parameters
    @param state_dict: The original model weights
    @param orig_lvls: Trained corruption levels
    @param new_lvls:  Inference corruption levels
    @return: Model weights with modified BN params
    """
    new_state_dict = {}
    for key in state_dict.keys():
        if 'bn_' in key:
            for sig in new_lvls:
                if len(orig_lvls) > 1:
                    lower, upper = getBounds(orig_lvls, sig)
                    if sig > upper:
                        alpha = 0
                        beta = 1
                    elif sig < lower:
                        alpha = 1
                        beta = 0
                    else:
                        alpha = (upper - sig) / (upper - lower)
                        beta = 1 - alpha

                    upper_key = key[:key.find('bn_') + 3] + str(upper) + key[key.rfind('.'):]
                    lower_key = key[:key.find('bn_') + 3] + str(lower) + key[key.rfind('.'):]
                    upper_tensor = state_dict[upper_key]
                    lower_tensor = state_dict[lower_key]
                    new_value = lower_tensor * alpha + beta * upper_tensor
                else:
                    new_value = state_dict[key]
                new_key = key[:key.find('bn_') + 3] + str(sig) + key[key.rfind('.'):]
                new_state_dict[new_key] = new_value
        else:
            new_state_dict[key] = state_dict[key]
    return new_state_dict


def main():
    global args
    args = parser.parse_args()

    # Create model
    model = HyperRes(meta_blocks=args.meta_blocks, level=args.lvls, device=args.device,
                     gray=args.y_channel, norm_factor=args.norm_f)

    # Load weights
    if not os.path.isfile(args.weights):
        print("=> no checkpoint found at '{}'".format(args.weights))
        exit()
    print("=> loading weights '{}'".format(args.weights))
    checkpoint = torch.load(args.weights, map_location=args.device)
    orig_sigmas = checkpoint['sigmas']
    new_weights = bn_interpolate(checkpoint['state_dict'], orig_sigmas, args.lvls)

    model.load_state_dict(new_weights)
    model.to(args.device)

    CommonTools.set_random_seed(0)
    torch.backends.cudnn.benckmark = True

    # Data loading code
    val_loader = NoisyDataset(
        os.path.join(args.data, args.valid),
        cor_lvls=args.lvls,
        phase='test',
        interp=False,
        lr_prefix=args.data_type,
    )
    val_loader = torch.utils.data.DataLoader(
        val_loader,
        batch_size=1, shuffle=False,
        num_workers=1, pin_memory=True)

    psnr_avg, lipips_avg = evaluate(val_loader, model)

    print("Results:")

    for k, v in psnr_avg.items():
        print("Sigma {}:\t {:4f}".format(k, v))

    print("LIPIPS:")

    for k, v in lipips_avg.items():
        print("Sigma {}:\t {:4f}".format(k, v))


def evaluate(val_loader, model):
    print("Evaluation:")

    sig_loss = [AverageMeter() for _ in range(len(args.lvls))]
    lpips_loss = [AverageMeter() for _ in range(len(args.lvls))]
    lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        random_image = np.random.randint(len(val_loader)) if args.save_sample else -1
        for i, data in enumerate(val_loader):
            target = data['target'].to(args.device)
            images = [x.to(args.device) for x in data['image']]

            if i == random_image:
                saveImage(output[j].detach().cpu().numpy()[0], target.detach().cpu().numpy()[0], j)

            # Compute output
            output = model(images)
            for j in range(len(args.lvls)):
                # Measure PSNR
                for out_idx, out in enumerate(output[j]):
                    imgs = postProcessForStats([target[out_idx], out])
                    trg, out = imgs
                    psnr = calculate_psnr(out, trg, args.data_type == 'sr')
                    lpips_loss[j].update(
                                        lpips(
                                                torch.FloatTensor(np.expand_dims(out.transpose((2,0,1)),0)/255.0),
                                                torch.FloatTensor(np.expand_dims(trg.transpose((2,0,1)),0)/255.0)
                                            )
                                        )
                    sig_loss[j].update(psnr)

    return {s: t.avg for s, t in zip(args.lvls, sig_loss)}, {s: t.avg for s, t in zip(args.lvls, lpips_loss)}


if __name__ == '__main__':
    main()
