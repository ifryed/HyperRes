import os

import cv2
import sys
from multiprocessing import Process

sys.path.append('../')

kSR_LEVELS = 1
kINPUT_FLD = 2
kOUTPUT_FLD = 3
kIMG_FILES = ['png', 'jpeg', 'jpg', 'bmp', 'tiff']


def genSR(sr_zoom, input_fld):
    output_fld = os.path.join(sys.argv[kOUTPUT_FLD], 'sr_{}'.format(sr_zoom))
    os.makedirs(output_fld, exist_ok=True)

    imgs_path_list = [x for x in os.listdir(input_fld) if x.split('.')[-1].lower() in kIMG_FILES]
    imgs_path_list.sort()

    for img_p in imgs_path_list:
        print('\rSR Level:{}, Img:{}'.format(sr_zoom, img_p), end='')

        img = cv2.imread(os.path.join(input_fld, img_p))
        h, w = img.shape[:2]
        img = cv2.resize(img, (0, 0), fx=1 / sr_zoom, fy=1 / sr_zoom, interpolation=cv2.INTER_CUBIC)
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)
        output_path = os.path.join(output_fld, img_p)
        cv2.imwrite(output_path, img)
    print()


def main():
    sr_levels = [int(x) for x in sys.argv[kSR_LEVELS].split(',')]
    input_fld = sys.argv[kINPUT_FLD]

    all_proc = []
    for sr_zoom in sr_levels:
        all_proc.append(Process(target=genSR, args=(sr_zoom, input_fld)))
        all_proc[-1].start()

    for p in all_proc:
        p.join()

    print("Done")


if __name__ == '__main__':
    if len(sys.argv) == 1:
        print("[SR levels(csv)] [input folder] [output folder]")
        exit()
    main()
