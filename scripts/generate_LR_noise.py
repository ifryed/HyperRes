import os
import numpy as np
import cv2
import sys
from multiprocessing import Process

kIMG_FILES = ['png', 'jpeg', 'jpg', 'bmp', 'tiff']


def genNoise(sigma, input_fld):
    output_fld = os.path.join(sys.argv[3], 'n_{}'.format(sigma))
    os.makedirs(output_fld, exist_ok=True)

    imgs_path_list = [x for x in os.listdir(input_fld) if x.split('.')[-1].lower() in kIMG_FILES]
    imgs_path_list.sort()

    for img_p in imgs_path_list:
        print('\rSigma:{}, Img:{}'.format(sigma, img_p), end='')

        img = cv2.imread(os.path.join(input_fld, img_p)) / 255
        h, w, c = img.shape
        noise_kernel = np.random.normal(0, sigma / 255, (h, w, c))

        n_img = img + noise_kernel
        n_img[n_img > 1] = 1
        n_img[n_img < 0] = 0
        n_img = (n_img * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(output_fld, img_p), n_img)
    print()


def main():
    sigmas = [int(x) for x in sys.argv[1].split(',')]
    input_fld = sys.argv[2]

    all_proc = []
    for sigma in sigmas:
        all_proc.append(Process(target=genNoise,args=(sigma, input_fld)))
        all_proc[-1].start()

    for p in all_proc:
        p.join()

    print("Done")


if __name__ == '__main__':
    if len(sys.argv) == 1:
        print("[Sigma levels(csv)] [input folder] [output folder]")
        exit()
    main()
