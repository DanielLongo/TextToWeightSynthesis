import torch

def reconstruction_loss(input_text, reconstructed_text):
	input_text = input_text.float().detach()
	reconstructed_text = reconstructed_text.float().detach()
	loss = torch.nn.functional.binary_cross_entropy(reconstructed_text, input_text)
	out = loss
	return out
