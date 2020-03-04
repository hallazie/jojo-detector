# coding:utf-8
#
# model by torch

import torch

from torch import nn
from torch.nn import functional as F
from torch.autograd import Function

class FrontalFaceClassifier(nn.Module):
	'''
		takes 256*144 (16*16, 9*16) size input
	'''
	def __init__(self):
		super().__init__()
		self.seq = nn.Sequential(
				nn.Conv2d(3, 32, kernel_size=1, stride=1, padding=0, groups=1, bias=False),
				nn.BatchNorm2d(32),
				nn.ReLU6(inplace=INPLACE),
				nn.MaxPool2d(2, stride=2), # 128*72
				nn.Conv2d(32, 64, kernel_size=1, stride=1, padding=0, groups=1, bias=False),
				nn.BatchNorm2d(64),
				nn.ReLU6(inplace=INPLACE),
				nn.MaxPool2d(2, stride=2), # 64*36
				nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0, groups=1, bias=False),
				nn.BatchNorm2d(128),
				nn.ReLU6(inplace=INPLACE),
				nn.MaxPool2d(2, stride=2), # 32*18
				nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0, groups=1, bias=False),
				nn.BatchNorm2d(128),
				nn.ReLU6(inplace=INPLACE),
				nn.MaxPool2d(2, stride=2), # 16*9
				nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0, groups=1, bias=False),
				nn.BatchNorm2d(128),
				nn.ReLU6(inplace=INPLACE),
				nn.MaxPool2d(2, stride=2), # 8*4
				nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0, groups=1, bias=False),
				nn.BatchNorm2d(128),
				nn.ReLU6(inplace=INPLACE),
				nn.MaxPool2d(4, stride=4), # 4*2
			)
		self.fc = nn.Linear(256, 1)

	def forward(self, x):
		x = self.seq(x)
		x = self.fc(x)
		return x