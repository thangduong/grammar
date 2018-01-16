from framework.utils.data.text_indexer import TextIndexer
from framework.evaluator import Evaluator
import framework.utils.common as utils
import os
import numpy as np
import time
import pickle
import math
import sys
from tokenex.tokenizer import Tokenizer
run_server = False
rtime = time.time()

prelim = False
infile = None
out_file = None

paramsfile = sys.argv[1]
params = utils.load_param_file(paramsfile)#'output/rnnlmV8/par# ams.py')
dir_path = os.path.dirname(paramsfile)

if not prelim:
	infile = '../we_test/devtest_sent_list.pkl'
	out_file= '../we_test/devtest_sent_prob.pkl'
#infile = '../we_test/valid_sent_list.pkl'
#out_file= '../we_test/valid_sent_prob.pkl'
	cmd = 'tar -czvf ' + str(rtime) + '.tgz ' + dir_path
	print(cmd)
	os.system(cmd)
vocab_file = os.path.join(utils.get_dict_value(params,'output_location'), 'vocab.pkl')
ckpt = os.path.join(utils.get_dict_value(params,'output_location'),
										utils.get_dict_value(params, 'model_name') + '.ckpt')


e = Evaluator.load2(ckpt)
i = TextIndexer.from_file(vocab_file)
tok = Tokenizer(0)
num_before = utils.get_dict_value(params, "num_words_before")
num_after = utils.get_dict_value(params, "num_words_after")
pad_tok = utils.get_dict_value(params, "pad_tok", '<pad>')

#sentences = ['<s> The apple , which is rotten , is not edible .']
sentences = ["I ate a pair of orange ."
							,'I ate a pear of orange .'
							,'I ate a pair of oranges .'
						 ]
cell_size = params['cell_size']
num_steps = params['num_steps']
num_layers = params['num_layers']
if infile is not None and os.path.exists(infile) and not prelim:
	with open(infile,'rb') as f:
		sentences = pickle.load(f)

all_sentence_prob = {}
unk_list = []
for sentence in sentences:
	orig_sentence = sentence
	sentence = sentence
#	sentence = sentence.replace("\u20a9 s","'s")
#	sentence = sentence.replace("’ s", "'s")
	sentence = sentence.lower()
	state = np.zeros([num_layers, 2, 1, cell_size])
	tokens = ["<s>"]+tok.tokenize(sentence)
	#print(tokens)
	sentence_len = len(tokens)
#	print(tokens)
#	print(sentence_len)
	sentence_prob = 1.0
	before = time.time()
	ntoks = len(tokens)
	if (len(tokens)%num_steps) > 0:
		tokens += ['.'] * (num_steps-(len(tokens)%num_steps))
	tokens += ['.']
	_,indexed,num_unk,unk_words = i.index_wordlist(tokens)
	unk_list += unk_words
#	print(indexed)
	for toki in range((int(len(indexed)/num_steps))):
		r = e.eval({'x': 
[indexed[(toki*num_steps):(toki+1)*num_steps]],
				"smei":
[indexed[(toki*num_steps+1):((toki+1)*num_steps+1)]],
								'state': state}, ['output_single_sm', 'final_state'])
		state = r[1]
#		print("output_single_sm:")
#		print(r[2])
		for j in range(num_steps):
			if toki*num_steps+j >= sentence_len-1:
				break
#			print(tokens[j+toki*20])
			#print(r[0][0][j][indexed[toki*20+j]])
#			wprob = r[0][0][j][nw]
			wprob = r[0][0][j]
#			print("%s ---- %s"%(wprob,wprob2))
			sentence_prob *= wprob
#			sm = np.argmax(r[0][0][j])
			nw = indexed[toki*num_steps+j+1]
#			print("%s %s - %s"%(nw, i._index_to_word[nw],wprob))
#			print("%s: %s: %s, %s, %s"%(sentence_prob,wprob,tokens[toki*num_steps+j],i._index_to_word[sm],wprob))
	diff = time.time() - before
	print("execute time = %s, unk_count = %s %s" % (diff, num_unk, unk_words))
#	print("%s:%s:%s"%(sentence_prob, math.log(sentence_prob), sentence))
	recval = math.log(sentence_prob)/ntoks
	print("%s,%s:%s"%(sentence_prob,  recval, tokens))
	all_sentence_prob[orig_sentence] = recval
#math.log(sentence_prob)/ntoks

if out_file is not None:
	with open(out_file, 'wb') as f:
		pickle.dump(all_sentence_prob,f)

#print(unk_list)
