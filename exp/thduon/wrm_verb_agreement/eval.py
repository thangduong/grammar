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

gflags.DEFINE_string('paramsfile', 'params.py', 'parameter files')
FLAGS = gflags.FLAGS


def split_sentence_for_eval(sentence, keywords, num_before, num_after):
	result = []
	sentence = ['<pad>']*num_before + sentence + ['<pad>']*num_after
	# TODO: multiple word tok
	for toki,tok in enumerate(sentence):
		if tok in keywords:
			result.append((sentence[(toki-num_before):(toki+num_after)],toki-num_before))
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
	keywords = params['keywords']
	rkeywords = params['id_to_keyword']
	vocab_file = os.path.join(utils.get_dict_value(params, 'output_location'), 'vocab.pkl')
	e = Evaluator.load2(ckpt)
	i = TextIndexer.from_file(vocab_file)
	e.dump_variable_sizes()
	exit(0)
	test_sentence = "it is better to die happy then to live miserably"
#	test_sentence = "there going two make a turkey today"
#	test_sentence = "they 're going two make a turkey today"
	test_sentence = "to big to fail my ass !"
#	test_sentence = "two big is not bad"
#	test_sentence = "<S> two big fishes in the same bucket"
#	test_sentence = "<S> there are two fishes in the same bucket ."
#	test_sentence = "<S> there are too fishes in the same bucket ."
	#test_sentence = "<S> I had two fishes for dinner"
	test_sentence = "<S> its raining men . hallelujah !"
#	test_sentence = "it 's head is too big"
	#test_sentence = "if it 's not one than it 's true"
	#test_sentence = "i would except it , but i don 't think it 's true"

	print("HELLO")
	print(keywords)
	split_sentence = list(split_sentence_for_eval(test_sentence.split(), keywords, num_before, num_after))
	print(split_sentence)
	for j in range(len(split_sentence)):
		print(split_sentence[j][0])
		_, sentence, _, _ = i.index_wordlist(split_sentence[j][0])
		bef = time()
		r = e.eval({'sentence': [sentence]}, {'sm_decision'})
		aft = time()
		#print(r[0][0])
		sm = r[0][0]
		am = np.argmax(sm)
		#print(am)
		k = rkeywords
		if am == 0:
			print("DO NOTHING")
		else:
			print(k[am - 1])
		k = [''] + k
		sm, k = zip(*sorted(zip(sm, k), reverse=True))
	#	print(k)
		for q,(x,y) in enumerate(zip(sm,k)):
			if q > 10:
				break
			print("%0.4f %s" %(x,y))
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
