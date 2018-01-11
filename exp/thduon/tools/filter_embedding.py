from framework.utils.data.word_embedding import WordEmbedding
import json
import pickle
import os

vocab_filename = '/mnt/work/training_data.tok2/1b/vocab/merged_vocab.txt'
embedding_filename = '../we_test/glove.pkl'
output_filename = "glove_small.txt"
#embedding_filename = '../we_test/GoogleNews-vectors-negative300.pkl'
#output_filename = "word2vec_small.txt"

glove = WordEmbedding.from_file(embedding_filename)
#word2vec = WordEmbedding.from_file('GoogleNews-vectors-negative300.pkl')
with open(vocab_filename, 'rb') as f, open(output_filename, "w") as of:
	for line in f:
		line = line.rstrip().lstrip()
		pieces = line.split()
		word = pieces[0].decode('utf-8')
		v = glove.lookup_word(word)
		if v is not None:
			v2 = [str(x) for x in v]

			of.write("%s %s\n"%(word, " ".join(v2)))
#		else:
#			print(word)