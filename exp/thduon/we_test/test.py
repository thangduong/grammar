from framework.utils.data.word_embedding import WordEmbedding
import math
import json
import pickle
import os

dataset = 'devtest'


if dataset == 'valid':
	output_data_file = 'valid_txt.txt'
	input_path = '../../../NeuralRewriting/eval/uhrs_dec_2017/valid.json'
	output_path = '../../../NeuralRewriting/eval/valid.json'
	sentence_list_path = 'valid_sent_list.pkl'
	sentence_prob_path = 'valid_sent_prob.pkl'
elif dataset == 'devtest':
	output_data_file = 'devtest_txt.txt'
	input_path = '../../../NeuralRewriting/eval/uhrs_dec_2017/devtest.json'
	output_path = '../../../NeuralRewriting/eval/devtest.json'
	sentence_list_path = 'devtest_sent_list.pkl'
	sentence_prob_path = 'devtest_sent_prob.pkl'
elif dataset == 'test':
	output_data_file = 'test_txt.txt'
	input_path = '../../../NeuralRewriting/eval/uhrs_dec_2017/test.json'
	output_path = '../../../NeuralRewriting/eval/test.json'
	sentence_list_path = 'test_sent_list.pkl'
	sentence_prob_path = 'test_sent_prob.pkl'

use_tok_sent = False
use_sent_prob = True


#input_path = '/mnt/work/NeuralRewriting.old/eval/small_eval_data.json'
#output_path = '/mnt/work/NeuralRewriting.old/eval/small_eval_data_out.json'
#output_data_file = "small_eval_data.txt"
#sentence_list_path = 'sent_list.pkl'
#sentence_prob_path ='sent_prob.pkl'
#use_tok_sent = False

#input_path = '/mnt/work/NeuralRewriting/eval/small_eval_data.json'
#output_path = '/mnt/work/NeuralRewriting/eval/small_eval_data_out.json'
#output_data_file = "small_eval_data.txt"
#sentence_prob_path = 'sent_prob.pkl'
#sentence_list_path = 'sent_list.pkl'
#use_tok_sent = False

# load embeddings
glove = WordEmbedding.from_file('glove.pkl')
word2vec = WordEmbedding.from_file('GoogleNews-vectors-negative300.pkl')

# load result from large LM if available
sent_prob = None
if sentence_prob_path is not None and os.path.exists(sentence_prob_path) and use_sent_prob:
	print("Loading cached sentence probability")
	with open(sentence_prob_path, 'rb') as f:
		sent_prob = pickle.load(f)

# input data
with open(input_path) as f:
	data = json.load(f)

unk_count = 0
sent_list = []
glove_unk_list = []
word2vec_unk_list = []
if output_data_file is not None:
	f = open(output_data_file, 'w')
else:
	f = None
for e in data:
	if f is not None:
		f.write('%s\n'% e['orig_sent'])
	query_word = e['query_word']
	options = e['options']
	if f is not None:
		f.write('%s\n'% query_word)
	for o in options:
		if f is not None:
			f.write('%s\n'% o['sent'])
		if use_tok_sent:
			sent_list.append(o['tok_sent'])
		else:
			sent_list.append(o['sent'])
		synonym = o['synonym']
		if f is not None:
			f.write('%s\n'% synonym)
#		print(sent_list[-1])

		# from glove
		glove_sim = glove.word_similarity(query_word, synonym)
		o['glove'] = glove_sim
		if glove_sim==-1:
			glove_unk_list.append(synonym)
			unk_count += 1

		# from word2vec
		word2vec_sim = word2vec.word_similarity(query_word, synonym)
		o['word2vec'] = word2vec_sim
		if word2vec_sim == -1:
			word2vec_unk_list.append(synonym)
			unk_count += 1

		# from large LM, if available
		if sent_prob is not None:
			sp = sent_prob[o['sent']]
#			if sp <= 0:
#				sp = -10
			print("sp = %s\n"%sp)
			o['large_lm'] = sp
			if word2vec_sim > 0 and glove_sim > 0:
				emb = word2vec_sim * .5 + glove_sim * .5
			else:
				emb = 0
			o['emb'] = emb
			for k in range(0, 100):
				ens = (1 - (k / 100)) * (sp / 10) + (k / 100) * emb
				o[str(k)] = ens

		for k in range(0, 101):
			if word2vec_sim > 0 and glove_sim > 0:
				o["e"+str(k)] = word2vec_sim * (1-(k/100)) + glove_sim * (k/100)
			else:
				if word2vec_sim > 0:
					o["e"+str(k)] = word2vec_sim
				else:
					o["e"+str(k)] = glove_sim

if f is not None:
	f.close()

# save output
with open(output_path,'w') as f:
	json.dump(data,f)

with open(sentence_list_path, 'wb') as f:
	pickle.dump(sent_list, f)

print("There are %s unknown synonyms for GloVe" % len(glove_unk_list))
print("There are %s unknown synonyms for Word2Vec" % len(word2vec_unk_list))
print("UNK COUNT = %s"%unk_count)
