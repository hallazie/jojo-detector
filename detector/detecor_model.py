# coding:utf-8
#
# model by torch

import torch
import torch.nn as nn

class ConvBlock(nn.Module):
	def __init__(self, cin, cout, ksize):
		self.conv = nn.Sequential(
			nn.Conv2d(cin, cout, kernel_size=ksize)
			nn.BatchNorm(cout)
			nn.ReLU(inplace=True)
		)

	def forward(self, x):
		x = self.conv(x)
		return x

class PoolBlock(nn.Module):
	def __init__(self, cin, cout, ksize):
		self.pool = nn.Sequential(
			nn.MaxPool2d(2, stride=2)
		)

	def forward(self, x):
		x = self.pool(x)
		return x	

class YoloFace(nn.Module):
	def __init__(self):
		super(YoloFace, self).__init__()
		pass

	def forward(self, x):
		return x

class Model(nn.Module):
	def __init__(self):
		super(Model, self).__init__()

		self.backbone = nn.Sequential(
			ConvBlock(3, 64, 3),
			ConvBlock(64, 64, 3),
			PoolBlock(),
			ConvBlock(64, 128, 3),
			ConvBlock(128, 128, 3),
			PoolBlock(),
			ConvBlock(128, 256, 3),
			ConvBlock(256, 256, 3),
			PoolBlock(),
			ConvBlock(256, 512, 3)
			ConvBlock(512, 512, 3)
			PoolBlock(),
			ConvBlock(512, 512, 3)
			ConvBlock(512, 512, 3)
			PoolBlock(),
			ConvBlock(512, 5, 1)
			YoloFace()
			)

	def nms(self, x):
		pass

	def forward(self, x):
		pass

