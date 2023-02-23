import os

import cv2
import argparse

import torch
import numpy as np

from models import HyperRes, NoiseNet
from utils.DataUtils.CommonTools import modcrop, ToTensor, calculate_psnr, postProcessForStats

alpha_slider_max = 150
title_window = 'MoNet LiveDemo'
min_v = 15
max_v = 75

device = "cuda:0"


def addNoise(src1, noise=None):
    global title_window
    sig = np.random.randint(0, 100)

    if noise:
        sig = noise
    title_window = "Noise Sigma {}".format(sig)
    ret_img = src1 + np.random.normal(0, sig / 255, src1.shape)
    return np.clip(ret_img, 0, 1), sig


def addSR(src1, lvl=None):
    global title_window

    if not lvl:
        lvl = np.random.randint(2, 6)
    title_window = "SR Factor {}".format(lvl)
    ret_img = cv2.resize(src1, (0, 0), fx=1 / lvl, fy=1 / lvl, interpolation=cv2.INTER_CUBIC)
    ret_img = cv2.resize(ret_img, tuple(src1.shape[-2::-1]), interpolation=cv2.INTER_CUBIC)
    return np.clip(ret_img, 0, 1), lvl * 10


def addDeJPEG(src1, lvl=None):
    global title_window

    if not lvl:
        lvl = np.random.randint(10, 100)
    title_window = "SR Factor {}".format(lvl)
    cv2.imwrite('tmp.jpg', src1 * 255, [int(cv2.IMWRITE_JPEG_QUALITY), lvl])
    ret_img = cv2.imread('tmp.jpg', 0) / 255
    os.remove('tmp.jpg')
    return np.clip(ret_img, 0, 1), lvl


def loadNoiseNet(device):
    weight_path = 'pre_trained/noise_detect.pth'
    if not os.path.exists(weight_path):
        print("No weights for NoiseNet")
        return None

    noise_model = NoiseNet().to(device)
    noise_model.load_state_dict(torch.load(weight_path, map_location=device), strict=False)
    return noise_model


def main():
    parser = argparse.ArgumentParser(description='Code for Adding a Trackbar to our applications tutorial.')
    parser.add_argument('--input',
                        help='Path to example image, if it\'s a folder, a random image will be chosen from that folder',
                        required=True)
    parser.add_argument('--checkpoint', help='Path to the model weighs.', required=True)
    parser.add_argument('--meta_blocks', type=int, default=16, help='Number of Meta Blocks')
    parser.add_argument('-y', '-Y', '--gray', dest='y_channel', default=False, action='store_true',
                        help='Train on Grayscale only')
    parser.add_argument('--data_type', type=str, default='n', choices=['n', 'sr', 'j'],
                        help='Defines the task data, de(n)oise, super-resolution(sr), de(j)peg.')

    args = parser.parse_args()
    img_path = args.input
    curr_dict = {"sr": addSR, "n": addNoise, "j": addDeJPEG}
    if os.path.isdir(args.input):
        img_path = os.path.join(args.input, np.random.choice(os.listdir(args.input)))
    if args.data_type == "j":
        gt = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    else:
        gt = cv2.imread(img_path)

    gt = modcrop(gt, 2)

    alpha_slider_max = 150
    if args.data_type == "sr":
        alpha_slider_max = 6 * 10
    if args.data_type == "j":
        alpha_slider_max = 100

    corrupt_fun = curr_dict[args.data_type]
    src1, rnd_noise = corrupt_fun(gt / 255)

    h, w = gt.shape[:2]
    canvas = np.zeros((h, 2 * w, 3)).astype(np.uint8)
    if args.data_type == "j":
        canvas = np.zeros((h, 2 * w)).astype(np.uint8)
    disp_src = src1
    if args.data_type == 'sr':
        disp_src = cv2.resize(src1, (0, 0), fx=10 / rnd_noise, fy=10 / rnd_noise)
    d_h, d_w = disp_src.shape[:2]
    canvas[:d_h, :d_w] = disp_src * 255
    src1 = ToTensor()(src1).to(device)

    # Create model
    checkpoint = torch.load(args.checkpoint, map_location='cuda:0')

    model = HyperRes(meta_blocks=args.meta_blocks,
                     level=[0], device=device,
                     gray=args.data_type == 'j').to(device)
    model.load_state_dict(checkpoint['state_dict'])

    noise_net = loadNoiseNet(device) 
    init_noise = 45
    if noise_net:
        init_noise = noise_net(src1.unsqueeze(0))
        print("Noise in image:\t{}\nPredicted Noise:\t{:.2f}".format(rnd_noise, init_noise.item()))
        print("==================================\n")

    def on_trackbar(alpha):
        alpha = alpha / 10 if args.data_type == 'sr' else alpha
        model.setLevel(alpha)
        with torch.no_grad():
            dst = model([src1.unsqueeze(0)])[0]
        dst = postProcessForStats(dst)[0]
        print("{} : {:.2f}".format(alpha, calculate_psnr(dst, gt)))
        canvas[:, w:] = dst
        cv2.imshow(title_window, canvas)

    cv2.namedWindow(title_window)
    cv2.imshow(title_window, canvas)
    trackbar_name = "Alpha"
    cv2.createTrackbar(trackbar_name, title_window, init_noise, alpha_slider_max, on_trackbar)
    # on_trackbar(init_noise)

    # Wait until user press some key
    cv2.waitKey()


if __name__ == '__main__':
    main()
