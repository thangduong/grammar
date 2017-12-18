from framework.utils.data.text_indexer import TextIndexer
from framework.evaluator import Evaluator
import framework.utils.common as utils
import word_classifier.data as data
import numpy as np
import math
from time import time
import pickle
import gflags
import os
import sys
import json

gflags.DEFINE_string('paramsfile', 'output/clmtV1/params.py', 'parameter files')
FLAGS = gflags.FLAGS


def split_sentence_for_eval(sentence, keywords, num_before, num_after):
	result = data._gen_data(None, sentence, keywords, num_before=num_before, num_after=num_after,
                          ignore_negative_data=True, add_redundant_keyword_data=False)
	return result

def merge_sentence(sentence, num_before, num_after, mid_word):
	before = sentence[:num_before]
	after = sentence[num_after:]
	rlist = (before + [mid_word] + after)
	rlist = [x for x in rlist if x!="<pad>"]
	result = ' '.join(rlist)
	return result

def eval(params,
				 save_accuracy_file=True,
				 batch_size=5000,
				 num_batches=20,
				 topn=1,
				 verbose=True):
	num_before = utils.get_dict_value(params, "num_words_before")
	num_after = utils.get_dict_value(params, "num_words_after")
	ckpt = os.path.join(utils.get_dict_value(params,'output_location'),
											utils.get_dict_value(params, 'model_name') + '.ckpt')
	accuracy_file = os.path.join(utils.get_dict_value(params,'output_location'),
											'accuracy.txt')
	vocab_file = os.path.join(utils.get_dict_value(params, 'output_location'), 'vocab.pkl')
	keywords_file = os.path.join(utils.get_dict_value(params, 'output_location'), 'keywords.pkl')
	e = Evaluator.load2(ckpt)
	i = TextIndexer.from_file(vocab_file)
	#test_sentence = "<S> ___ quick brown fox jumped over the lazy dog"
	test_sentence = "<S> ___ is no way to know whether it will work"
	#test_sentence = "<S> ___ house is on fire"
#	test_sentence = "<S> ___ in your best interest to lie"
#	test_sentence = "<S> ___ yours and I cannot touch it"
	#test_sentence = "<S> I ate a ___ and an apple"
	#test_sentence = "<S> I have to take ___ life away"
#	test_sentence = "<S> ___ may and it is raining"
	#test_sentence = "<S> This will take ___ before it will actually work"
	#test_sentence = "<S> this is probably bigger ___ that"
#	test_sentence = "<S> ___ is no place like home"
	#test_sentence = "I have ___ of money"
	#test_sentence = "<S> I think I ___ have it"
	test_sentence = "<S> don 't forget to get orange , banana , and ___ ."
#	test_sentence = "<S> in the heat ___ the night"
#	test_sentence = "<S> in the river , ___ the boat"
#	test_sentence = "<S> nothing can be ___ from the truth"
#	test_sentence = "<S> the ___ knot will unwind"
#	test_sentence = "<S> if you keep playing, you will ___ ."
	test_sentence = "<s> I ate a ___ of oranges ."
#	test_sentence = "<s> I ate a ___ and oranges ."
#	test_sentence = "<s> I live in a ___ ."
#	test_sentence = "<s> I ate a ___ of oranges ."
	test_sentence = "<s> I ate a ___ and oranges ."
	test_sentence = "<s> I live in a ___ ."
	test_sentence = "<s> I have seen it on him , and can ___ to it ."
	test_sentence = "<s> the thieves ___ the library and got very little for their pains ."

	# input data
	with open('/mnt/work/NeuralRewriting/eval/small_eval_data.json') as f:
		data = json.load(f)
	with open(keywords_file, 'rb') as f:
		k = pickle.load(f)

	unk_list = []
	for q in data:
		query_word = q['query_word']
		orig_sent = q['orig_sent']
		options = q['options']
		orig_sent = orig_sent.replace(query_word, "___")
		orig_sent = "<s> " + orig_sent
		test_sentence = orig_sent.lower()
		split_sentence = list(split_sentence_for_eval(test_sentence.split(), ["___"], num_before, num_after))
#		print(split_sentence[0][0])
		_, sentence, _, _ = i.index_wordlist(split_sentence[0][0])
		bef = time()
		r = e.eval({'sentence': [sentence]}, {'sm_decision'})
		aft = time()
		sm = r[0][0]

		for o in options:
			synonym = o['synonym']
			if synonym not in k:
				score = -1
				unk_list += [synonym]
			else:
				score = -math.log(sm[k.index(synonym)])
			o['clmtV1'] = score
			print(score)

	# save output
	with open('/mnt/work/NeuralRewriting/eval/small_eval_data_out.json','w') as f:
		json.dump(data,f)

	print(len(unk_list))
def main(argv):
	try:
		argv = FLAGS(argv)  # parse flags
	except gflags.FlagsError as e:
		print('%s\\nUsage: %s ARGS\\n%s' % (e, sys.argv[0], FLAGS))
		sys.exit(1)
	print(FLAGS.paramsfile)
	params = utils.load_param_file(FLAGS.paramsfile)
	eval(params,
				 save_accuracy_file=False,
				 batch_size=10000,
				 num_batches=20,
				 topn=5)


if __name__ == '__main__':
	main(sys.argv)
