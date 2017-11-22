from framework.utils.data.text_indexer import TextIndexer
from framework.evaluator import Evaluator
import framework.utils.common as utils
import word_classifier.data as data
import copy
import numpy as np
from time import time
import pickle
import gflags
import os
import sys


def split_sentence_for_eval(tokens, keywords, num_before, num_after):
	pad_tok = "<pad>"
	tokens = [pad_tok] * (num_before -1) + ['<S>']+ tokens + [pad_tok]*(num_after+5)
	result = []
	for toki in range(num_before, len(tokens)-num_before-5):
		tok0 = tokens[toki]
		tok1 = tuple([x for x in tokens[toki:toki + 2]])
		tok2 = tuple([x for x in tokens[toki:toki + 3]])
		bucket = None
		sl = 0
		if tok2 in keywords:
			bucket = keywords[tok2]
			sl = 3
		elif tok1 in keywords:
			bucket = keywords[tok1]
			sl = 2
		elif tok0 in keywords:
			bucket = keywords[tok0]
			sl = 1
		if bucket is None:
			None
		else:
			task_input = tokens[toki-num_before:toki+num_after]
			before_index = toki - num_before
			after_index = toki + sl - num_before
			result.append([task_input, before_index, after_index])
	return result

class WRMEval:
	def __init__(self):
		None

	def load(self, model_dir):
		self._model_dir = model_dir
		self._paramsfile = os.path.join(self._model_dir, 'params.py')
		self._params = utils.load_param_file(self._paramsfile)
		self._num_before = utils.get_dict_value(self._params, "num_words_before")
		self._num_after = utils.get_dict_value(self._params, "num_words_after")
		ckpt = os.path.join(utils.get_dict_value(self._params,'output_location'),
												utils.get_dict_value(self._params, 'model_name') + '.ckpt')
		vocab_file = os.path.join(utils.get_dict_value(self._params, 'output_location'), 'vocab.pkl')
		self._e = Evaluator.load2(ckpt)
		self._i = TextIndexer.from_file(vocab_file)
		self._keywords = self._params['keywords']
		self._id_to_word = self._params['id_to_keyword']

	def critique(self, sentence):
		tokens = sentence.split()
		print(tokens)
		tasks = split_sentence_for_eval(tokens, self._keywords, self._num_before, self._num_after)
		corrections = []
		for task in tasks:
			print(task[0])
			_,it,_,_ = self._i.index_wordlist(task[0])
			r = self._e.eval({'sentence': [it]}, {'sm_decision'})
			sm = r[0][0]
			repl = np.argmax(sm)
			pr = sm[repl]
			if repl > 0 and pr > 0.6:
				target_word = self._id_to_word[repl-1]
				src_word = tokens[task[1]:task[2]]
				corrections.append([task[1], task[2], src_word, target_word])

			k = [''] + self._id_to_word
			sm, k = zip(*sorted(zip(sm, k), reverse=True))
			for q, (x, y) in enumerate(zip(sm, k)):
				if q > 10:
					break
				print("%0.4f %s" % (x, y))
		return corrections, tokens

	def markup_critique(self, corrections, tokens):
		result = ""
		i = 0
		cur_corrections = copy.deepcopy(corrections)
		while (i < len(tokens)): # can escape early  if we run out of critiques
			if len(result)>0:
				result += ' '
			if len(cur_corrections)>0 and i >= cur_corrections[0][0]:
				i = cur_corrections[0][1]
				src_word = cur_corrections[0][2]
				if type(cur_corrections[0][3]) is not tuple:
					target_word = [cur_corrections[0][3]]
				else:
					target_word = list(cur_corrections[0][3])
				target_word = ' '.join(target_word)
				src_word = ' '.join(src_word)
				result += "<b><strike>%s</strike>%s</b>" % (src_word,target_word)
				del cur_corrections[0]
			else:
				result += tokens[i]
				i += 1
		return result

	def get_model_name(self):
		return utils.get_dict_value(self._params, 'model_name', '_UNKNOWN_MODEL_')


if __name__ == '__main__':
	e = WRMEval()
	e.load("./output/wrmV1/")
	sentence = "whatever man , i think this is two big for me too deal with"
	corrections, tokens = e.critique(sentence)
	markup = e.markup_critique(corrections, tokens)
	print(markup)
