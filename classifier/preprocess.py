# coding:utf-8

import random
import os

from shutil import copyfile

def gen_dataset():
	root_raw = 'G:/datasets/face/jojo/frames'
	root_new = 'G:/datasets/face/jojo/clf'
	for _,_,fs in os.walk(root_raw):
		random.shuffle(fs)
		for f in fs[:2000]:
			if not f.endswith('jpg'):
				continue
			path_raw = '%s/%s' % (root_raw, f)
			path_new = '%s/%s' % (root_new, f)
			print('%s -> %s' % (path_raw, path_new))
			copyfile(path_raw, path_new)

if __name__ == '__main__':
	gen_dataset()