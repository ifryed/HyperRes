import os
import cv2
import sys
from multiprocessing import Process

kJPEG_LEVELS = 1
kINPUT_FLD = 2
kOUTPUT_FLD = 3
kIMG_FILES = ['png', 'jpeg', 'jpg', 'bmp', 'tiff']


def genDEJPEG(jpg_quality, input_fld):
    output_fld = os.path.join(sys.argv[kOUTPUT_FLD], 'j_{}'.format(jpg_quality))
    os.makedirs(output_fld, exist_ok=True)

    imgs_path_list = [x for x in os.listdir(input_fld) if x.split('.')[-1].lower() in kIMG_FILES]
    imgs_path_list.sort()

    for img_p in imgs_path_list:
        print('\rJPEG Level:{}, Img:{}'.format(jpg_quality, img_p), end='')

        img = cv2.imread(os.path.join(input_fld, img_p), cv2.IMREAD_UNCHANGED)
        output_path = os.path.join(output_fld, img_p)
        output_path = output_path[:output_path.rfind('.png')] + ".jpg"
        cv2.imwrite(output_path, img,
                    [int(cv2.IMWRITE_JPEG_QUALITY), jpg_quality])
    print()


def main():
    print("DeJPEG generator")
    jpeg_levels = [int(x) for x in sys.argv[kJPEG_LEVELS].split(',')]
    input_fld = sys.argv[kINPUT_FLD]

    all_proc = []
    for jpeg_quality in jpeg_levels:
        all_proc.append(Process(target=genDEJPEG, args=(jpeg_quality, input_fld)))
        all_proc[-1].start()

    for p in all_proc:
        p.join()

    print("Done")


if __name__ == '__main__':
    if len(sys.argv) == 1:
        print("[JPEG levels(csv)] [input folder] [output folder]")
        exit()
    main()
