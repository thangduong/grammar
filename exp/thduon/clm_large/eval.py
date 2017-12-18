from framework.utils.data.text_indexer import TextIndexer
from framework.evaluator import Evaluator
import framework.utils.common as utils
import word_classifier.data as data
import numpy as np
from time import time
import pickle
import gflags
import os
import sys

gflags.DEFINE_string('paramsfile', 'output/clmV0/params.py', 'parameter files')
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
	test_sentence = test_sentence.lower()
	split_sentence = list(split_sentence_for_eval(test_sentence.split(), ["___"], num_before, num_after))
#	print(split_sentence[0][0])
	print(split_sentence[0][0])
	_, sentence, _, _ = i.index_wordlist(split_sentence[0][0])
	bef = time()
	r = e.eval({'sentence': [sentence]}, {'sm_decision'})
	aft = time()
	#print(r[0][0])
	sm = r[0][0]
	am = np.argmax(sm)
	with open(keywords_file, 'rb') as f:
		k = pickle.load(f)
	#print(am)
#	if am == 0:
#		print("DO NOTHING")
#	else:
#		print(k[am - 1])
	k = k
	sm, k = zip(*sorted(zip(sm, k), reverse=True))
#	print(k)
	wlist = ['migrate','contribute','swear','write','climb']
	wlist = ['ruled', 'ransacked', 'identified', 'visited', 'enjoyed']
	for x in wlist:
		print("%s %s"%(x,sm[k.index(x)]))
	for i,(x,y) in enumerate(zip(sm,k)):
		if i>20:
			break
		print("%d %f %s" %(i, x,y))
	print(test_sentence)
	print("EVAL TIME = %s"%(aft-bef))

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
