from __future__ import print_function, division
import os
import random
import numpy as np
from torch.utils.data import Dataset
from .CommonTools import ToTensor, augment, read_img, modcrop

kIMG_EXT = ['png', 'jpg', 'jpeg', 'bmp', 'tiff']


def im_crop(img, y, x, y_crp, w_crp):
    return img[y:y + y_crp, x: x + w_crp]


class NoisyDataset(Dataset):
    """Corrupted images dataset"""

    """
        os.path.join(args.data, 'train'),
        cor_lvls=args.lvls,
        crop_size=crop_size,
        phase='train', lr_prefix=args.data_type)"""

    def __init__(self, root_dir: str, cor_lvls: list, crop_size: int = 128, phase: str = 'train', interp: bool = False,
                 lr_prefix: str = 'n'):
        """
        Initializes the Dataset
        @param root_dir: The folder that holdes the dataset (test,train,valid)
        @param cor_lvls: The corruption levels to load. I.E. for Noise 15,75
        @param crop_size: Cropping the image to crop_size X crop_size
        @param phase: Train or Test
        @param interp: If False, loads the corrupted images from the disk,
                        other-wise creates the corrupted images from the GT
        @param lr_prefix: If interp is False, what is the prefix for the folder
                            that contains the  corrupted levels images
        """
        self.root_dir_gt = os.path.join(root_dir, 'clean')
        self.root_dir = root_dir
        self.images_lst = [x for x in os.listdir(self.root_dir_gt) if x[x.rfind('.') + 1:] in kIMG_EXT]
        self.crop_size = crop_size
        self.transform = ToTensor()
        self.cor_lvls = cor_lvls
        self.images = dict()
        self.phase = phase
        self.interp = interp
        self.LR_fld_prefix = lr_prefix
        if self.interp:
            self.lr_ext=""
        else:
            valid_img_sample = os.listdir(os.path.join(self.root_dir, "{}_{}".format(self.LR_fld_prefix, cor_lvls[0])))[0]
            self.lr_ext = valid_img_sample[valid_img_sample.find('.'):]

    def __len__(self):
        return len(self.images_lst)

    def __getitem__(self, idx):
        trg_img_name = self.images_lst[idx]
        target = read_img(os.path.join(self.root_dir_gt, trg_img_name))
        
        if self.interp:
            # Corrupting the images 'On the fly'
            # TODO: Works for Gaussian noise only, need to add more or delete the option
            n_targets = []
            target = modcrop(target, 2)
            if self.phase == 'train':
                h, w = target.shape[:2]
                x, y = random.randint(0, w - self.crop_size), random.randint(0, h - self.crop_size)
                target = im_crop(target, y, x, self.crop_size,self.crop_size)
            else:
                h, w = target.shape[:2]
                x, y = random.randint(0, w - self.crop_size), random.randint(0, h - self.crop_size)
                target = im_crop(target, y, x, self.crop_size,self.crop_size)
            for gam in self.cor_lvls:
                tmp_n = (target + np.random.normal(0, gam / 255, target.shape))
                tmp_n = self.transform(tmp_n)
                tmp_n = np.clip(tmp_n, 0, 1)
                n_targets.append(tmp_n)
            target = self.transform(target)
        else:
            trg_img_name = self.images_lst[idx]
            n_targets = []
            trg_img_name = trg_img_name[:trg_img_name.rfind('.')] + self.lr_ext
            for sigma in self.cor_lvls:
                n_targets.append(
                    read_img(
                        os.path.join(self.root_dir, "{}_{}".format(self.LR_fld_prefix, sigma), trg_img_name))
                )

            if self.phase == 'train':
                h, w = target.shape[:2]
                # Crop
                x, y = 0, 0
                w_crp, y_crp = h, w
                if not self.crop_size is None:
                    x, y = random.randint(0, w - self.crop_size), random.randint(0, h - self.crop_size)
                    w_crp = y_crp = self.crop_size

                for idx, img in enumerate(n_targets):
                    n_targets[idx] = im_crop(img, y, x, y_crp, w_crp)
                target = im_crop(target, y, x, y_crp, w_crp)

                all_imgs = augment([target, *n_targets], hflip=True, rot=True)
                target = self.transform(all_imgs[0])
                n_targets = [self.transform(x) for x in all_imgs[1:]]
            elif self.phase != 'train':
                target = self.transform(modcrop(target, 2))
                n_targets = [self.transform(modcrop(x, 2)) for x in n_targets]
                
        sample = {'image': n_targets, 'target': target}
        return sample
