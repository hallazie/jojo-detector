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
	epochs = 100
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
	dataloader = iter(torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True))
	for e in range(epochs):
		for i in range(len(dataloader)):
			image, label = next(dataloader)
			predict = model(image.to(device))
			loss = criterion(predict.to(device), label.to(device).float()).to(device)
			print('epoch %s, batch %s, BCELoss = %s' % (e, i, loss.data.item()))
			loss.backward()
			optimizer.step()
			optimizer.zero_grad()
		if e % (epochs//10) == 0:
			torch.save(model, os.path.join('model', 'model_%s.pkl' % (e)))

if __name__ == '__main__':
	train()