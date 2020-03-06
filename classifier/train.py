# coding:utf-8

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import os

# from tqdm import tqdm
from torchvision import transforms, utils
from frontal_classifier_model import FrontalFaceClassifier

def train():
	epochs = 500
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model = FrontalFaceClassifier().to(device)
	criterion = nn.BCELoss()
	optimizer = optim.Adam(model.parameters(), lr=1e-3)
	dataset = torchvision.datasets.ImageFolder('G:/datasets/face/jojo/clf',
												transform=transforms.Compose([
													transforms.Resize((256, 144)),
													transforms.ToTensor(),
													transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
													])
												)
	dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)
	dataiter = iter(dataloader)
	image, label = next(dataiter)
	for e in range(epochs):
		for i in range(len(dataiter)):
			try:
				predict = model(image.to(device))
				loss = criterion(predict.to(device), label.to(device).float()).to(device)
				print('epoch %s, batch %s, BCELoss = %s' % (e, i, loss.data.item()))
				image, label = next(dataiter)
				loss.backward()
				optimizer.step()
				optimizer.zero_grad()
			except StopIteration as sie:
				dataiter = iter(torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True))
				image, label = next(dataiter)
		if e % ((epochs+1)//50) == 0:
			torch.save(model, os.path.join('../model', 'clf-model-%s.pkl' % (e)))

if __name__ == '__main__':
	train()