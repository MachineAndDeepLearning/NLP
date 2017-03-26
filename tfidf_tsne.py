import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from util import get_wikipedia_data, init_weight
from sklearn.feature_extraction.text import TfidfTransformer


def find_analogies(w1, w2, w3, we_file='word_embeddings.npy', w2i_file='wikipedia_word2idx.json'):
	We = np.load(we_file)
	with open(w2i_file) as f:
		word2idx = json.load(f)

	king = We[word2idx[w1]]
	man = We[word2idx[w2]]
	woman = We[word2idx[w3]]
	v0 = king - man + woman

	def dist1(a, b):
		return np.linalg.norm(a - b)

	def dist2(a, b):
		return 1 - a.dot(b) / (np.linalg.norm(a) * np.linalg.norm(b))

	for dist, name in [(dist1, 'Euclidean'), (dist2, 'cosine')]:
		min_dist = float('inf')
		best_word = ''
		for word, idx in word2idx.iteritems():
			if word not in (w1, w2, w3):
				v1 = We[idx]
				d = dist(v0, v1)
				if d < min_dist:
					min_dist = d
					best_word = word
		print("closest match by", name, "distance:", best_word)
		print(w1, "-", w2, "=", best_word, "-", w3)


def main():
	sentences, word2idx = get_wikipedia_data(n_files=10, n_vocab=1500, by_paragraph=True)
	with open('w2v_word2idx.json', 'w') as f:
		json.dump(word2idx, f)

	# build term document matrix
	V = len(word2idx)
	N = len(sentences)

	# create raw counts first
	A = np.zeros((V, N))
	j = 0
	for sentence in sentences:
		for i in sentence:
			A[i, j] += 1
		j += 1
	print("finished getting raw counts")

	transformer = TfidfTransformer()
	A = transformer.fit_transform(A)
	# print "type(A):", type(A)
	# exit()
	A = A.toarray()

	idx2word = {v: k for k, v in word2idx.items()}

	# plot the data in 2-D
	tsne = TSNE()
	Z = tsne.fit_transform(A)
	plt.scatter(Z[:, 0], Z[:, 1])
	for i in range(V):
		try:
			plt.annotate(s=idx2word[i].encode("utf8"), xy=(Z[i, 0], Z[i, 1]))
		except:
			print("bad string:", idx2word[i])
	plt.show()

	# create a higher-D word embedding, try word analogies
	# tsne = TSNE(n_components=3)
	# We = tsne.fit_transform(A)
	We = Z
	find_analogies('king', 'man', 'woman', We, word2idx)
	find_analogies('france', 'paris', 'london', We, word2idx)
	find_analogies('france', 'paris', 'rome', We, word2idx)
	find_analogies('paris', 'france', 'italy', We, word2idx)


if __name__ == '__main__':
	main()
