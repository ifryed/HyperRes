from multiprocessing.sharedctypes import Value
import os
import os.path
import sys
from multiprocessing import Pool
import numpy as np
import cv2
import sys
from shutil import get_terminal_size
import time

class ProgressBar(object):
    '''A progress bar which can print the progress
    modified from https://github.com/hellock/cvbase/blob/master/cvbase/progress.py
    '''

    def __init__(self, task_num=0, bar_width=50, start=True):
        self.task_num = task_num
        max_bar_width = self._get_max_bar_width()
        self.bar_width = (bar_width if bar_width <= max_bar_width else max_bar_width)
        self.completed = 0
        if start:
            self.start()

    def _get_max_bar_width(self):
        terminal_width, _ = get_terminal_size()
        max_bar_width = min(int(terminal_width * 0.6), terminal_width - 50)
        if max_bar_width < 10:
            print('terminal width is too small ({}), please consider widen the terminal for better '
                  'progressbar visualization'.format(terminal_width))
            max_bar_width = 10
        return max_bar_width

    def start(self):
        if self.task_num > 0:
            sys.stdout.write('[{}] 0/{}, elapsed: 0s, ETA:\n{}\n'.format(
                ' ' * self.bar_width, self.task_num, 'Start...'))
        else:
            sys.stdout.write('completed: 0, elapsed: 0s')
        sys.stdout.flush()
        self.start_time = time.time()

    def update(self, msg='In progress...'):
        self.completed += 1
        elapsed = time.time() - self.start_time
        fps = self.completed / elapsed
        if self.task_num > 0:
            percentage = self.completed / float(self.task_num)
            eta = int(elapsed * (1 - percentage) / percentage + 0.5)
            mark_width = int(self.bar_width * percentage)
            bar_chars = '█' * mark_width + '-' * (self.bar_width - mark_width)
            sys.stdout.write('\033[2F')  # cursor up 2 lines
            sys.stdout.write('\033[J')  # clean the output (remove extra chars since last display)
            sys.stdout.write('[{}] {}/{}, {:.1f} task/s, elapsed: {}s, ETA: {:5}s\n{}\n'.format(
                bar_chars, self.completed, self.task_num, fps, int(elapsed + 0.5), eta, msg))
        else:
            sys.stdout.write('completed: {}, elapsed: {}s, {:.1f} tasks/s'.format(
                self.completed, int(elapsed + 0.5), fps))
        sys.stdout.flush()

def main():
	input_folder = sys.argv[1]
	save_folder = sys.argv[2]
	n_thread = 20
	crop_sz = 480
	step = 240
	thres_sz = 48
	compression_level = 0
	
	os.makedirs(save_folder,exist_ok=True)
	print('mkdir [{:s}] ...'.format(save_folder))
	img_list = []
	for root, _, file_list in sorted(os.walk(input_folder)):
		path = [os.path.join(root,x) for x in file_list]
		img_list.extend(path)
	print(len(img_list))

	def update(arg):
		pbar.update(arg)

	pbar = ProgressBar(len(img_list))
	pool = Pool(n_thread)
	for path in img_list:
		pool.apply_async(worker,
			args=(path,save_folder,crop_sz,step,thres_sz, compression_level),
			callback=update)

	pool.close()
	pool.join()
	print("All is Done")

def worker(path,save_folder,crop_sz,step,thres_sz,compression_level):
	img_name=os.path.basename(path)
	img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

	n_channels = len(img.shape)
	if n_channels == 2:
		h,w = img.shape
	elif n_channels == 3:
		h,w,c = img.shape
	else:
		raise ValueError('Wrong image shape - {}'.format(n_channels))

	h_space = np.arange(0, h-crop_sz+1,step)
	if h - (h_space[-1]+crop_sz) > thres_sz:
		h_space = np.append(h_space, h - crop_sz)
	w_space = np.arange(0, w-crop_sz + 1,step)
	if w - (w_space[-1]+crop_sz) > thres_sz:
		w_space = np.append(w_space, w - crop_sz)

	index = 0
	for x in h_space:
		for y in w_space:
			index +=1
			if n_channels == 2:
				crop_img = img[x:x+crop_sz, y:y+crop_sz]
			else:
				crop_img = img[x:x+crop_sz, y:y+crop_sz,:]
			crop_img = np.ascontiguousarray(crop_img)

			cv2.imwrite(
				os.path.join(save_folder, img_name.replace('.png','_s{:03d}.png'.format(index))),
				crop_img, [cv2.IMWRITE_PNG_COMPRESSION, compression_level]
			)
	return 'Processing {:s} ...'.format(img_name)

if __name__ == '__main__':
	main()

