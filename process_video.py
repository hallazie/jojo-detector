# coding:utf-8

from PIL import Image
from tqdm import tqdm

import cv2
import imagehash
import os

class Processer():
	def __init__(self):
		self.clfpath = 'F:/env/Python37/Lib/site-packages/cv2/data/haarcascade_frontalface_alt2.xml'
		# self.root = 'G:/episodes/[异域-11番小队][JoJo的奇妙冒险JoJo_no_Kimyou_na_Bouken][BDRIP][1-26+SP][X264-10bit_AAC][720P]'
		# self.root = 'G:/episode s/JoJo no Kimyou na Bouken BD-1080p-asxzwang'
		# self.root = 'G:/episodes/JoJo\'s Bizarre Adventure Stardust Crusaders Battle in Egypt BD-1080p-asxzwang'
		self.root = 'G:/episodes/[JOJO&UHA-WING&Kamigami][JoJo\'s Bizarre Adventure - Golden Wind][01-39][x264 1080p][CHT]'
		self.savepath = 'G:/datasets/face/jojo/frame'
		self.rectcolor = (228, 16, 16)

	def _init_data(self):
		self.clf = cv2.CascadeClassifier(self.clfpath)

	def _save_all(self):
		for _,_,fs in os.walk(self.root):
			for i, f in enumerate(fs):
				print(f'processing {i}th file {f}')
				if not f.endswith('mp4'):
					continue
				vpath = os.path.join(self.root, f)
				print(vpath)
				self._save_frame(vpath, i)
				print(f'{f} finished')

	def _gen_video(self):
		fps = 24
		size = (662, 332)
		videowriter = cv2.VideoWriter('output/gan-sample.mp4', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, size)
		path = 'F:/machinelearning/vision/style-based-gan-pytorch/sample/'
		for i, name in enumerate(os.listdir(path)):
			if i % 3 != 0:
				continue
			img = cv2.imread(path + name)
			img = cv2.resize(img, size)
			videowriter.write(img)


	def _save_frame(self, vpath, vid):
		vidcap = cv2.VideoCapture(vpath)
		success, image = vidcap.read()
		success = True
		prevhash = None
		pbar = tqdm(range(24*60*24))
		for idx in pbar:
			saveflag = True
			success, image = vidcap.read()
			if not success:
				break
			if idx % int(24*0.5) == 0:
				currhash = imagehash.average_hash(Image.fromarray(image.astype('uint8')))
				if prevhash is not None:
					diff = prevhash - currhash
					if diff < 25:
						saveflag = False
				prevhash = currhash
				# grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
				# rectlist = self.clf.detectMultiScale(grey, scaleFactor=1.2, minNeighbors=15, minSize=(32, 32), flags=4)
				# if len(rectlist) > 0:
				# 	for rect in rectlist:
				# 		x, y, w, h = rect
				# 		cv2.rectangle(image, (x - 10, y - 10), (x + w, y + h), self.rectcolor, 2)
				if saveflag:
					cv2.imwrite(f'G:/datasets/face/jojo/frames/5-frame-{vid}-{str(idx).zfill(6)}.jpg', image)

	@staticmethod
	def rename():
		root = 'G:/datasets/face/jojo/1&2'
		flist = os.listdir(root)
		for f in flist:
			if not f.endswith('jpg'):
				continue
			t = '1-' + f
			old = root + '/' + f
			new = root + '/' + t
			os.rename(old, new)

if __name__ == '__main__':
	processer = Processer()
	processer._init_data()
	processer._save_all()
	# processer.rename()
	# processer._gen_video()