# coding:utf-8

from PIL import Image
from tqdm import tqdm

import cv2
import imagehash

class Processer():
	def __init__(self):
		self.clfpath = 'F:/env/Python37/Lib/site-packages/cv2/data/haarcascade_frontalface_alt2.xml'
		self.rectcolor = (228, 16, 16)
		self._init_data()

	def _init_data(self):
		self.clf = cv2.CascadeClassifier(self.clfpath)

	def _eval(self):
		# vidcap = cv2.VideoCapture('G:/episodes/[异域-11番小队][JoJo的奇妙冒险JoJo_no_Kimyou_na_Bouken][BDRIP][1-26+SP][X264-10bit_AAC][720P]/[YYDM-11FANS][JoJo_no_Kimyou_na_Bouken][01][X264-10bit_AAC][720P][1AA45D3C].mp4')
		vidcap = cv2.VideoCapture('data/jojo-1.mp4')
		success, image = vidcap.read()
		success = True
		prevhash = None
		pbar = tqdm(range(24*60*24))
		for idx in pbar:
			saveflag = True
			success, image = vidcap.read()
			if not success:
				break
			if idx % 24*1 == 0:
				currhash = hash = imagehash.average_hash(Image.fromarray(image.astype('uint8')))
				if prevhash is not None:
					diff = prevhash - currhash
					if diff < 25:
						saveflag = False
				prevhash = currhash
				grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
				rectlist = self.clf.detectMultiScale(grey, scaleFactor=1.2, minNeighbors=15, minSize=(32, 32), flags=4)
				if len(rectlist) > 0:
					for rect in rectlist:
						x, y, w, h = rect
						cv2.rectangle(image, (x - 10, y - 10), (x + w, y + h), self.rectcolor, 2)
				if saveflag:
					cv2.imwrite(f'output/frame-{str(idx).zfill(6)}.jpg', image)
			# if idx > 9999:
			# 	break
			idx += 1

if __name__ == '__main__':
	processer = Processer()
	processer._eval()