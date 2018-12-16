import torch
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import CountVectorizer

def hash_vectorize(text, n=20):
	vectorizer = HashingVectorizer(n_features=n)
	vector = vectorizer.transform(text).toarray()
	vector = torch.from_numpy(vector).float()
	return vector

def count_vectorize(text):
	vectorizer = CountVectorizer()
	vectorizer.fit(text)
	vector = vectorizer.transform(text).toarray()
	vector = torch.from_numpy(vector).float()
	return vector
	
