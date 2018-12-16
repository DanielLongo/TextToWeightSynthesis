import torch
from torch import nn

class WeightSynthesizer(nn.Module):
	def __init__(self, num_dims_in, num_dims_out, d=25):
		super(WeightSynthesizer, self).__init__()
		self.fc1 = nn.Sequential(
			nn.Linear(num_dims_in, d),
			nn.ReLU()
		)
		self.fc2 = nn.Sequential(
			nn.Linear(d, d*2),
			nn.ReLU()
		)
		self.fc3 = nn.Sequential(
			nn.Linear(d*2, d*4),
			nn.ReLU()
		)
		self.fc4 = nn.Sequential(
			nn.Linear(d*4, num_dims_out),
			nn.Sigmoid()
		)

	def forward(self, x):
		out = self.fc1(x)
		out = self.fc2(out)
		out = self.fc3(out)
		out = self.fc4(out)
		return out
