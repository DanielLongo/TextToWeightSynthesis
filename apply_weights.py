import torch
from torch import nn

class ApplyWeights(nn.Module):
	def __init__(self, images_true, images_false, weight_dim):
		super(ApplyWeights, self).__init__()
		self.images_true = images_true
		self.images_false = images_false
		self.fc = nn.Linear(1, weight_dim)
		self.weight_dim = weight_dim

	def forward(self, weights):
		self.fc = nn.Linear(weights.shape[0], weights.shape[1])
		# print(self.fc.weight.shape)
		weights = torch.transpose(weights, 0 , 1)
		self.fc.weight = nn.Parameter(weights)
		true_sum = 0
		false_sum = 0
		for image_true, image_false in zip(self.images_true, self.images_false):
			image_true = image_true.view(-1, 1).repeat(1, weights.shape[1])
			image_false = image_false.view(-1, 1).repeat(1, weights.shape[1])
			# print("image_true", image_true.shape)
			# print("out", self.fc(image_true).shape)
			true_sum += torch.sum(self.fc(image_true))
			false_sum += torch.sum(self.fc(image_false))

		return true_sum, false_sum

		
