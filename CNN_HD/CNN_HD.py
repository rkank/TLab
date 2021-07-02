# %%

import sys
import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.nn as nn
from torchvision import transforms, datasets
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
from tqdm.notebook import trange, tqdm
import matplotlib.pyplot as plt
import ray

sys.path.insert(1, 'C:\\Users\\ruchi\\Desktop\\Research\\TLab\\HyperDimensional\\Tools')

import HDComputing as hdc

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ray.init()

NUM_EPOCHS = 10
batch_size = 1000


class HDCodec(nn.Module):
	def __init__(self):
		super().__init__()

		enc = {'record': {'N': 400, 'M': 50, 'range': (-0.25, 0.25)}}
		self.space = hdc.Hyperspace(rep=hdc.BSCVector, dim=10000, enc=enc)

		# self.spaces = []
		#
		# for epoch_num in range(10):
		# 	self.spaces.append(hdc.Hyperspace(rep=hdc.BSCVector, dim=10000, enc=enc))

	@ray.remote
	def __encode(self, features, label):
		self.space.add(name=label, features=features)
		return None

	def forward(self, x, batch_labels, epoch_num):
		feature_list = x.detach().cpu().numpy()

		# for num in range(feature_list.shape[0]):
		# 	label = str(batch_labels[num].item())
		# 	features = feature_list[num]
		#
		# 	space_num = 0
		# 	while space_num <= epoch_num:
		# 		self.spaces[space_num].add(name=label, features=features)
		# 		space_num += 1

		futures = [self.__encode.remote(self, feature_list[num], str(batch_labels[num].item())) for num in
		           range(feature_list.shape[0])]
		outputs = ray.get(futures)

		return x


class LeNet5(nn.Module):
	def __init__(self):
		super().__init__()

		self.convs = nn.Sequential(
			nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2, bias=True),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2),
			nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0, bias=True),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2))
		self.fc = nn.Sequential(
			nn.Linear(16 * 5 * 5, 120),
			nn.ReLU(),
			nn.Linear(120, 84),
			nn.ReLU(),
			nn.Linear(84, 10))

		self.codec = HDCodec()

	def forward(self, img, label, epoch_num):
		x = self.convs(img)
		x = x.view(-1, 16 * 5 * 5)
		x = self.codec(x, label, epoch_num)
		x = self.fc(x)

		return x


train_dataset = datasets.MNIST('../data', train=True, download=False,
                               transform=transforms.Compose([
	                               transforms.ToTensor(),
	                               transforms.Normalize((0.1307,), (0.3081,))
                               ]))

test_dataset = datasets.MNIST('../data', train=False, download=False,
                              transform=transforms.Compose([
	                              transforms.ToTensor(),
	                              transforms.Normalize((0.1307,), (0.3081,))
                              ]))

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

model = LeNet5().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

epoch_losses = []
model.train()

for epoch in trange(NUM_EPOCHS):
	epoch_loss = 0
	num_batches = 0

	for batch_num, (data, labels) in enumerate(tqdm(train_loader)):
		data, labels = data.to(device), labels.to(device)
		outputs = model(data, labels, epoch)
		batch_loss = criterion(outputs, labels)

		optimizer.zero_grad()
		batch_loss.backward()
		epoch_loss += batch_loss.detach().cpu().data
		num_batches += 1

		optimizer.step()

	scheduler.step(epoch_loss / num_batches)

	epoch_losses.append(epoch_loss / num_batches)

	torch.save({
		'epoch': epoch,
		'model_state_dict': model.state_dict(),
		'optimizer_state_dict': optimizer.state_dict(),
		'loss': epoch_losses[-1],
	}, 'model.ckpt')

	print("epoch: ", epoch, ", loss: ", epoch_loss / num_batches)
