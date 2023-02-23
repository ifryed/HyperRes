import os
import sys

import numpy as np
from skimage import io


def main():
    images = os.listdir(sys.argv[1])
    out_fld = sys.argv[2]
    pop_n = int(sys.argv[3]) if len(sys.argv) > 3 else 200

    img = io.imread(os.path.join(sys.argv[1], images[0]))
    h, w = img.shape[:2]
    crop_size = 256
    for i in range(pop_n):
        x, y = np.random.randint(0, w - crop_size), np.random.randint(0, h - crop_size)
        io.imsave(sys.argv[2] + '/img_{}.png'.format(i), img[y:y + crop_size, x: x + crop_size])


if __name__ == '__main__':
    main()
