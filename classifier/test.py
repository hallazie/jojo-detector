# coding:utf-8

import torch
import numpy as np
import os

from torchvision import transforms
from PIL import Image
from shutil import copyfile

def test():
	datapath = '../data'
	transform=transforms.Compose([
		transforms.Resize((256, 144)),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
		])
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model = torch.load('../model/clf-model-400.pkl').to(device)
	for _,_,fs in os.walk(datapath):
		for f in fs:
			if not f.endswith('.jpg'):
				continue
			path = os.path.join(datapath, f)
			data = transform(Image.open(path))
			output = model(torch.unsqueeze(data, 0).to(device)).detach()
			pred = output[0][0].cpu()
			print('[%s] pred: %s, has face: %s' % (f, pred, 'Yes' if pred > 0.5 else 'No'))
			save = '../output/%s-%s.jpg' % (f, 'Yes' if pred > 0.5 else 'No')
			Image.open(path).save(save)

def inference():
	datapath = 'G:/datasets/face/jojo/frames'
	transform=transforms.Compose([
		transforms.Resize((256, 144)),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
		])
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model = torch.load('../model/clf-model-400.pkl').to(device)
	cnt = 0
	for _,_,fs in os.walk(datapath):
		for i, f in enumerate(fs):
			if not f.endswith('.jpg'):
				continue
			path = os.path.join(datapath, f)
			data = transform(Image.open(path))
			output = model(torch.unsqueeze(data, 0).to(device)).detach()
			pred = output[0][0].cpu()
			if pred > 0.5:
				save = 'G:/datasets/face/jojo/face-raw/%s.jpg' % f
				copyfile(path, save)
				cnt += 1
			if i % 500 == 0:
				print('%sth finished, with face preserved: %s' % (i, cnt))

if __name__ == '__main__':
	inference()