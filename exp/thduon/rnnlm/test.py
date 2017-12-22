from framework.utils.data.word_embedding import WordEmbedding
import json
import pickle
import os

# load embeddings
if os.path.exists('glove.pkl'):
	glove = WordEmbedding.from_file('glove.pkl')
else:
	glove = None
if os.path.exists('GoogleNews-vectors-negative300.pkl'):
	word2vec = WordEmbedding.from_file('GoogleNews-vectors-negative300.pkl')
else:
	word2vec = None

# load result from large LM if available
sent_prob = None
if os.path.exists('sent_prob.pkl') or False:
	print("Loading cached sentence probability")
	with open('sent_prob.pkl', 'rb') as f:
		sent_prob = pickle.load(f)

# input data
with open('/mnt/work/NeuralRewriting/eval/small_eval_data.json') as f:
	data = json.load(f)

sent_list = []
glove_unk_list = []
word2vec_unk_list = []

for e in data:
	query_word = e['query_word']
	options = e['options']
	for o in options:
		sent_list.append(o['sent'])
		synonym = o['synonym']

		# from glove
		if glove is not None:
			glove_sim = glove.word_similarity(query_word, synonym)
			o['glove'] = glove_sim
			if glove_sim==-1:
				glove_unk_list.append(synonym)

		# from large LM, if available
		if sent_prob is not None:
			sp = sent_prob[o['sent']]
			o['lm'] = sp

		# from word2vec
		if word2vec is not None:
			word2vec_sim = word2vec.word_similarity(query_word, synonym)
			o['word2vec'] = word2vec_sim
			if word2vec_sim == -1:
				word2vec_unk_list.append(synonym)

# save output
with open('/mnt/work/NeuralRewriting/eval/small_eval_data_out.json','w') as f:
	json.dump(data,f)

with open('sent_list.pkl', 'wb') as f:
	pickle.dump(sent_list, f)

print("There are %s unknown synonyms for GloVe" % len(glove_unk_list))
print("There are %s unknown synonyms for Word2Vec" % len(word2vec_unk_list))
