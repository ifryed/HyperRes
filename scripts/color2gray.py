import os
import os.path
import sys
from multiprocessing import Pool
import cv2

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    """A multi-thread tool for converting RGB images to gray images."""

    input_folder = sys.argv[1]
    save_folder = sys.argv[2]
    compression_level = 9
    n_thread = 20  # thread number

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        print('mkdir [{:s}] ...'.format(save_folder))
    else:
        print('Folder [{:s}] already exists. Exit...'.format(save_folder))
        sys.exit(1)

    img_list = []
    for root, _, file_list in sorted(os.walk(input_folder)):
        path = [os.path.join(root, x) for x in file_list]  # assume only images in the input_folder
        img_list.extend(path)

    pool = Pool(n_thread)
    for path in img_list:
        pool.apply_async(worker, args=(path, save_folder, compression_level))
    pool.close()
    pool.join()
    print('All subprocesses done.')


def worker(path, save_folder, compression_level):
    img_name = os.path.basename(path)
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # BGR
    img_y = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(
        os.path.join(save_folder, img_name), img_y,
        [cv2.IMWRITE_PNG_COMPRESSION, compression_level])
    return 'Processing {:s} ...'.format(img_name)


if __name__ == '__main__':
    main()
