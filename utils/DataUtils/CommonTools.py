from typing import List, Tuple

import cv2
import torch
import numpy as np
import random
from torch.nn import init


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        if not type(sample) == np.ndarray:
            sample = np.array(sample)
        if sample.ndim == 2:
            sample = np.expand_dims(sample, 2)
        image = np.ascontiguousarray(sample.transpose((2, 0, 1)))
        return torch.from_numpy(image).float()


def perform_augment(d_hflip: bool, d_vflip: bool, d_rot: bool, img):
    """
    Performs the image augmentaion
    @param d_hflip: Do Horizontal flip
    @param d_vflip: Do Vertical flip
    @param d_rot: Do Rotate 90 degrees
    @param img:
    @return:
    """
    if d_hflip:
        img = img[:, ::-1, :]
    if d_vflip:
        img = img[::-1, :, :]
    if d_rot:
        img = img.transpose(1, 0, 2)
    return img


def augment(img_list: list, hflip: bool = True, rot: bool = True) -> List[np.ndarray]:
    """
    Augments the image inorder to add robustness to the model
    @param img_list: The List of images
    @param hflip: If True, add horizontal flip
    @param rot: If True, add 90 degrees rotation
    @return: A list of the augmented images
    """
    # horizontal flip OR rotate
    hflip = hflip and np.random.random() < 0.5
    vflip = rot and np.random.random() < 0.5
    rot90 = rot and np.random.random() < 0.5

    return [perform_augment(hflip, vflip, rot90, img) for img in img_list]


def modcrop(img_in: np.ndarray, scale: int) -> np.ndarray:
    """
    Cropes the images according to the scale
    @param img_in: Input image
    @param scale: The resizeing scale
    @return: The cropped image
    """
    # img_in: Numpy, HWC or HW
    img = np.copy(img_in)
    if img.ndim == 2:
        h, w = img.shape
        h_r, w_r = h % scale, w % scale
        img = img[:h - h_r, :w - w_r]
    elif img.ndim == 3:
        h, w, c = img.shape
        h_r, w_r = h % scale, w % scale
        img = img[:h - h_r, :w - w_r, :]
    else:
        raise ValueError('Wrong img dimensions: [{:d}].'.format(img.ndim))
    return img


def postProcessForStats(img_lst: List[np.ndarray], min_max: Tuple[float, float] = (0, 1)) -> List[np.ndarray]:
    """
    Performs post network process on the output to convert it to a normal RGB
    @param img_lst: The list of outputs from the network
    @param min_max: The min/max range to clip the output
    @return: A list of sRGB images, value range -> (0,255,dtype=np.uint8)
    """
    img_lst = [x.squeeze().float().cpu().clamp_(*min_max) for x in img_lst]
    img_lst = [(x - min_max[0]) / (min_max[1] - min_max[0]) for x in img_lst]
    n_dim = img_lst[0].dim()
    if n_dim == 2:
        img_lst = [x.numpy() for x in img_lst]
    elif n_dim == 3:
        img_lst = [np.transpose(x.numpy(), (1, 2, 0)) for x in img_lst]
    img_lst = [(x * 255).round().astype(np.uint8) for x in img_lst]

    return img_lst

def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2, on_gray=False):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')

def calculate_psnr(img1: np.ndarray, img2: np.ndarray, on_gray: bool = False) -> float:
    """
    Calculates the PSNR between img1 and img2. Both images values should should be (0,255)
    @param img1: Image 1
    @param img2: Image 2
    @param on_gray: If True, expecting the images to be gray scale
    @return: The PSNR -> [0,inf)
    """

    if on_gray:
        img1 = rgb2gray(img1)
        img2 = rgb2gray(img2)
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))


def read_img(path: str) -> np.ndarray:
    """
    Reads image, if it's a color image converts it to RGB. 
    @param path: Image paths
    @return: Image RGB/Gray
    """
    # return: Numpy float32, HWC, BGR, [0,1]
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    # img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    n_dim = img.ndim
    if n_dim == 2:
        img = np.expand_dims(img, 2)
    img = img.astype(np.float32) / 255.
    if n_dim == 2:
        return img
    return img[:, :, [2, 1, 0]]  # BGR to RGB

def bgr2ycbcr(img, only_y=True):
    '''bgr version of rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [24.966, 128.553, 65.481]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786],
                              [65.481, -37.797, 112.0]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)

def rgb2gray(img: np.ndarray) -> np.ndarray:
    """
    Convertes RGB to gray scale
    @param img: Input RGB
    @return: Gray scale image
    """
    return bgr2ycbcr(img[:, :, [2, 1, 0]], only_y=True)


def saveImage(output: np.ndarray, gt: np.ndarray, idx: int) -> np.ndarray:
    """
    Save the output of the network as an sRGB image
    @param output: The network output
    @param gt: The ground truth
    @param idx: Image idx
    @return: The saved image
    """
    output = np.clip(output, 0, 1)
    gt = np.clip(gt, 0, 1)
    output = output.squeeze()
    gt = gt.squeeze()
    n_dim = output.ndim
    if n_dim == 2:
        output = output * 255
        gt = gt * 255
    elif n_dim == 3:
        output = output.transpose((1, 2, 0)) * 255
        gt = gt.transpose((1, 2, 0)) * 255
    h, w = gt.shape[:2]
    canvas = np.zeros((h, 2 * w))
    if n_dim == 3:
        canvas = np.zeros((h, 2 * w, 3))
    canvas[:, :w] = output
    canvas[:, w:] = gt
    canvas = canvas.astype(np.uint8)

    cv2.imwrite('s_{:02d}.png'.format(idx), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    return canvas


def weights_init_kaiming(m, scale: float = 1):
    """
    Initializing the weights using the 'Kaiming' method
    @param m: The network layer
    @param scale: The sacale
    @return: None
    """
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.startswith('Conv'):
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1 and classname.find('BatchNorm2dParallel') < 0:
        init.constant_(m.weight.data, 1.0)
        init.constant_(m.bias.data, 0.0)


def set_random_seed(seed: int):
    """
    Sets the random seed on radnom,numpy and torch libraries
    @param seed: The seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def getBounds(sig_arr, n_sig):
    lower = sig_arr[0]
    upper = sig_arr[1]
    for i, v in enumerate(sig_arr[:-1]):
        if n_sig <= v:
            break
        lower = v
        upper = sig_arr[i + 1]

    return lower, upper


def bn_interpolate(state_dict, orig_sigmas, new_sigmas):
    new_state_dict = {}
    for key in state_dict.keys():
        if 'bn_' in key:
            for sig in [new_sigmas]:
                lower, upper = getBounds(orig_sigmas, sig)
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

                new_key = key[:key.find('bn_') + 3] + str(0) + key[key.rfind('.'):]
                new_state_dict[new_key] = lower_tensor * alpha + beta * upper_tensor
        else:
            new_state_dict[key] = state_dict[key]
    return new_state_dict
