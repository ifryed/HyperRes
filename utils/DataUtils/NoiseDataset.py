import glob

from torch.utils.data import Dataset
from PIL import Image
import numpy as np


class GenImageDataset(Dataset):
    def __init__(self, root_dir: str, phase: str = 'train', crop_size=128):
        self.root_dir = root_dir
        self.phase = phase

        self.images_path = glob.glob(self.root_dir + '/*.png')[:1000]
        self.crop_size = crop_size

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, idx):
        img = Image.open(self.images_path[idx])
        img = np.array(img, dtype=np.float32)

        # Crop image
        h, w = img.shape[:2]
        x, y = np.random.randint(0, w - self.crop_size), np.random.randint(0, h - self.crop_size)
        img = img[y:y + self.crop_size, x:x + self.crop_size, :]

        # Add Noise
        n_mean, n_var = 0, np.random.randint(0, 101)
        noise = np.random.normal(n_mean, n_var, img.shape)

        img_n = img + noise
        img_n = np.clip(img_n, a_min=0, a_max=255)/255

        noisy = np.transpose(img_n, (2, 0, 1)).astype(np.float32)
        trg = np.array(n_var, dtype=np.int64)
        return noisy, trg
