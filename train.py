# coding:utf-8

import torch
import torchvision
from torchvision import transforms, utils

def train():
	dataset = torchvision.datasets.ImageFolder('G:/datasets/face/jojo/clf',
												transform=transforms.Compose([
													transforms.Resize((256, 144)),
													transforms.ToTensor()])
												)
	dataloader = iter(torch.utils.data.DataLoader(dataset, batch_size=16,shuffle=True))
	batch = dataloader.__next__()
	print(batch[0].shape, batch[1].shape)


if __name__ == '__main__':
	train()