import torch
from text_vectorizer import count_vectorize
from weight_synthesizer import WeightSynthesizer
from apply_weights import ApplyWeights
from reconstruct_text import ReconstructText
from utils import reconstruction_loss
def test_text_encoder(): 
	text = ["there is green near bottom", "there are black and white spots"]
	v_text = count_vectorize(text)
	print("Vectorized Shape:", v_text.shape)
	return v_text

def test_weight_synthesizer(text):
	num_dims_in = text.shape[1]
	num_dims_out = 64
	ws = WeightSynthesizer(num_dims_in=num_dims_in, num_dims_out=num_dims_out)
	a = ws(text)
	print("Synthesized Weights:", a.shape)
	return a

def test_apply_weights(weights, images_true, images_false):
	tester = ApplyWeights(images_true, images_false, 64)
	score_true, score_false = tester(weights)
	print("Score true:", score_true)
	print("Score false:", score_false)

def test_reconstruction_loss(weights, text):
	reconstructer = ReconstructText(64, text.shape[1])
	wv = reconstructer(weights)
	wv.requies_grad = False
	text.requies_grad = False
	r_loss = reconstruction_loss(text, wv)
	print("Reconstruction Loss:", r_loss)

text = test_text_encoder()
weights  = test_weight_synthesizer(text)
images_true = [torch.ones(64), torch.ones(64), torch.ones(64)]
images_false = [torch.ones(64), torch.ones(64), torch.ones(64)]
test_apply_weights(weights, images_true, images_false)
test_reconstruction_loss(weights, text)