import torch
from torch import nn

class ReconstructText(nn.Module):
	def __init__(self, in_dim, out_dim, d=25):
		super(ReconstructText, self).__init__()
		self.fc1 = nn.Sequential(
			nn.Linear(in_dim, d),
			nn.ReLU()
		)
		self.fc2 = nn.Sequential(
			nn.Linear(d, d*2),
			nn.ReLU()
		)
		self.fc3 = nn.Sequential(
			nn.Linear(d*2, out_dim),
			nn.Sigmoid()
		)

	def forward(self, weights):
		out = self.fc1(weights)
		out = self.fc2(out)
		out = self.fc3(out)
		return out 