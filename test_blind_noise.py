"""
Training script for ImageNet
Copyright (c) Wei YANG, 2017
"""
from __future__ import print_function

import argparse
import os

import torch.nn.parallel
import torch.backends.cudnn as cudnn

import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import matplotlib.pyplot as plt

from models import HyperRes, NoiseNet

from utils import AverageMeter
from utils.DataUtils import CommonTools
from utils.DataUtils.CommonTools import calculate_psnr, postProcessForStats, saveImage, calculate_ssim
from utils.DataUtils.TrainLoader import NoisyDataset

parser = argparse.ArgumentParser(description='PyTorch ImageNet Testing')
parser.add_argument('-d', '--data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--weights', required=True, type=str, metavar='PATH',
                    help='path to latest checkpoint')
parser.add_argument('--meta_blocks', type=int, default=16,
                    help='Number of Meta Blocks')
parser.add_argument('--device', type=str, default='cpu',
                    help='Device to run on,[cpu,cuda..]')
parser.add_argument('--levels', type=int, nargs='+',
                    default=[15], help='input resolutions.')

def loadNoiseNet(device):
    noise_model = NoiseNet().to(device)
    noise_model.load_state_dict(torch.load('pre_trained/noise_detect.pth', map_location=device))
    return noise_model


def main():
    global args, best_prec1
    args = parser.parse_args()

    # Create model
    model = HyperRes(meta_blocks=args.meta_blocks, level=args.levels, device=args.device,
                     gray=False, norm_factor=255).to(args.device)
                     
    noise_model = loadNoiseNet(args.device)

    # Load weights
    if not os.path.isfile(args.weights):
        print("=> no checkpoint found at '{}'".format(args.weights))
        exit()
    print("=> loading weights '{}'".format(args.weights))
    checkpoint = torch.load(args.weights, map_location=args.device)

    model.load_state_dict(checkpoint['state_dict'])

    CommonTools.set_random_seed(42)
    torch.backends.cudnn.benckmark = True

    # Data loading code
    val_loader = NoisyDataset(
        args.data,
        cor_lvls=args.levels,
        phase='test',
        interp=False,
        lr_prefix='n',
    )
    val_loader = torch.utils.data.DataLoader(
        val_loader,
        batch_size=1, shuffle=False,
        num_workers=1, pin_memory=True)

    psnr_avg = validate(val_loader, model, noise_model)


def validate(val_loader, model: HyperRes, noise_model: NoiseNet):
    print("Validation:")

    sig_loss = [AverageMeter() for _ in range(len(args.levels))]
    ssim_loss = [AverageMeter() for _ in range(len(args.levels))]

    os.makedirs('blindNoise', exist_ok=True)

    model.eval()
    noise_model.eval()
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            target = data['target'].to(args.device)
            images = [x.to(args.device) for x in data['image']]

            print("=====================")
            print(i)
            print(args.levels)
            model.setLevel(args.levels)
            output = model(images)
            for j in range(len(args.levels)):
                # Measure PSNR
                for out_idx, out in enumerate(output[j]):
                    imgs = postProcessForStats([target[out_idx], out, images[j]])
                    trg, out, noise = imgs

                    plt.imsave("blindNoise/{}_gt.png".format(i), trg)
                    plt.imsave("blindNoise/{}_{}_noise.png".format(i, args.levels[j]), noise)
                    plt.imsave("blindNoise/{}_{}_semi.png".format(i, args.levels[j]), out)

                    psnr = calculate_psnr(out, trg, False)
                    ssim = calculate_ssim(out, trg, False)
                    sig_loss[j].update(psnr)
                    ssim_loss[j].update(ssim)
                    print("\t{:.3f}:\tPSNR:\t{:.3f}| SSIM\t{:.3f}".format(args.levels[j], psnr, ssim))

            # Blind
            new_sigmas = [noise_model(image) for image in images]
            model.setLevel(new_sigmas)
            output = model(images)

            new_sigmas = [x.detach().cpu().numpy()[0][0] for x in new_sigmas]
            print(new_sigmas)
            for j in range(len(new_sigmas)):
                # Measure PSNR
                for out_idx, out in enumerate(output[j]):
                    imgs = postProcessForStats([target[out_idx], out])
                    trg, out = imgs

                    plt.imsave("blindNoise/{:}_{:.3f}.png".format(i, new_sigmas[j]), out)

                    psnr = calculate_psnr(out, trg, False)
                    ssim = calculate_ssim(out, trg, False)
                    sig_loss[j].update(psnr)
                    ssim_loss[j].update(ssim)
                    print("\t{:.3f}:\tPSNR:\t{:.3f} SSIM\t{:.3f}".format(new_sigmas[j], psnr, ssim))

    model.train()

    return {s: (t.avg, ssim.avg) for s, t, ssim in zip(args.levels, sig_loss, ssim_loss)}


if __name__ == '__main__':
    main()
