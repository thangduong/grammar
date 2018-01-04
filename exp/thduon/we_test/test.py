from framework.utils.data.word_embedding import WordEmbedding
import json
import pickle
import os

input_path = '/mnt/work/NeuralRewriting.new/eval/uhrs_dec_2017/devtest.json'
output_path = '/mnt/work/NeuralRewriting.new/eval/uhrs_dec_2017/output.json'
output_data_file = 'devtest.txt'
sentence_list_path = 'sent_list.pkl'
sentence_prob_path = 'sent_prob.pkl'
use_tok_sent = False

#input_path = '/mnt/work/NeuralRewriting.old/eval/small_eval_data.json'
#output_path = '/mnt/work/NeuralRewriting.old/eval/small_eval_data_out.json'
#output_data_file = "small_eval_data.txt"
#sentence_list_path = 'sent_list.pkl'
#sentence_prob_path ='sent_prob.pkl'
#use_tok_sent = False

# load embeddings
glove = WordEmbedding.from_file('glove.pkl')
word2vec = WordEmbedding.from_file('GoogleNews-vectors-negative300.pkl')

# load result from large LM if available
sent_prob = None
if sentence_prob_path is not None and os.path.exists(sentence_prob_path):
	print("Loading cached sentence probability")
	with open(sentence_prob_path, 'rb') as f:
		sent_prob = pickle.load(f)

# input data
with open(input_path) as f:
	data = json.load(f)

sent_list = []
glove_unk_list = []
word2vec_unk_list = []
f = open(output_data_file, 'w')
for e in data:
	f.write('%s\n'% e['orig_sent'])
	query_word = e['query_word']
	options = e['options']
	f.write('%s\n'% query_word)
	for o in options:
		f.write('%s\n'% o['sent'])
		if use_tok_sent:
			sent_list.append(o['tok_sent'])
		else:
			sent_list.append(o['sent'])
		synonym = o['synonym']
		f.write('%s\n'% synonym)
#		print(sent_list[-1])

		# from glove
		glove_sim = glove.word_similarity(query_word, synonym)
		o['glove'] = glove_sim
		if glove_sim==-1:
			glove_unk_list.append(synonym)

		# from large LM, if available
		if sent_prob is not None:
			sp = sent_prob[o['sent']]
			o['large_lm'] = sp

		# from word2vec
		word2vec_sim = word2vec.word_similarity(query_word, synonym)
		o['word2vec'] = word2vec_sim
		if word2vec_sim == -1:
			word2vec_unk_list.append(synonym)

		o['e2'] = word2vec_sim + glove_sim * .04
f.close()

# save output
with open(output_path,'w') as f:
	json.dump(data,f)

with open(sentence_list_path, 'wb') as f:
	pickle.dump(sent_list, f)

print("There are %s unknown synonyms for GloVe" % len(glove_unk_list))
print("There are %s unknown synonyms for Word2Vec" % len(word2vec_unk_list))
