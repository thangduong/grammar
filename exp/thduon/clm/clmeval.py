from framework.utils.data.text_indexer import TextIndexer
from framework.evaluator import Evaluator
import framework.utils.common as utils
import copy
import numpy as np
from time import time
import pickle
import gflags
import os
import sys

def gen_data(tokens, keywords,
						 num_before=5, num_after=5,
						 pad_tok="<pad>", null_sample_factor=0,
						 add_redundant_keyword_data=True,
						 use_negative_only_data=True,
						 ignore_negative_data=False,
						 add_keyword_removal_data=False):
	tokens = [pad_tok] * (num_before-1) + ["<S>"] + tokens + [pad_tok]*(num_after+5)
	class_offset = 1
	if ignore_negative_data:
		class_offset = 0

	results = []
	unk_list = []
	no_insert_list = []
	for toki in range(num_before, len(tokens)-num_before-4):
		tok0 = tokens[toki].lower()
		tok1 = [x.lower() for x in tokens[toki:toki+2]]
		tok2 = [x.lower() for x in tokens[toki:toki+3]]
		if tok2 in keywords:
			ki = keywords.index(tok2)
			results.append(\
				(tokens[(toki-num_before):toki]+tokens[(toki+3):(toki+num_after+3)], \
				ki + class_offset, toki-num_before, toki-num_before+3))
		elif tok1 in keywords:
			ki = keywords.index(tok1)
			results.append(\
				(tokens[(toki-num_before):toki]+tokens[(toki+2):(toki+num_after+2)], \
				ki + class_offset, toki-num_before, toki-num_before+2))
		elif tok0 in keywords:
			ki = keywords.index(tok0)
			results.append( \
				(tokens[(toki - num_before):toki] + tokens[(toki + 1):(toki + num_after + 1)], \
				 ki + class_offset, toki-num_before, toki-num_before+1))
		else:
			# add unk
			if 'unk' in keywords:
				ki = keywords.index('unk')
				unk_list.append((tokens[(toki-num_before):toki]+tokens[(toki+1):(toki+num_after+1)], ki + class_offset))
		no_insert_list.append((tokens[(toki-num_before):toki]+tokens[(toki):(toki+num_after)], 0))
	return results


def gen_keywords(params):
	buckets = params['keyword_buckets']
	keywords_map = {}
	keywords_list = []
	idx = 0
	for bucket in buckets:
		for keyword in bucket:
			if type(keyword) is tuple:
				keywords_list.append(list(keyword))
			else:
				keywords_list.append(keyword)
			keywords_map[keyword] = copy.deepcopy(bucket)
			idx += 1
	return keywords_map, keywords_list


def split_sentence_for_eval(sentence, keywords, num_before, num_after):
	result = gen_data(sentence, keywords, num_before=num_before, num_after=num_after,
                          ignore_negative_data=True, add_redundant_keyword_data=False)
	return result

class CLMEval:
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
		with open(os.path.join(
				utils.get_dict_value(self._params, 'output_location'),
				'keywords.pkl'), 'rb') as f:
			keywords = pickle.load(f)
		self._params['keywords'] = keywords
		self._keywords = self._params['keywords']
		self._keyword_map, self._keyword_list = gen_keywords(self._params)

	def fill_in_the_blank(self, sentence, blank="___"):
		if type(sentence) is not list:
			result = list(split_sentence_for_eval(sentence.split(), [blank], self._num_before, self._num_after))
			sentence = result[0][0]
		_, sentence, _, _ = self._i.index_wordlist(sentence)
		bef = time()
		r = self._e.eval({'sentence': [sentence]}, {'sm_decision'})
		aft = time()

		sm = r[0][0]
		k = copy.deepcopy(self._keywords)
		sm, k = zip(*sorted(zip(sm, k), reverse=False))
#		print(k)
#		for x, y in zip(sm, k):
#			print("%0.4f %s" % (x, y))
		return sm,k

	def lookup_prob(self, sentence, proposed, blank="___"):
		sm, k = self.fill_in_the_blank(sentence, blank)
		result = []
		maxword = ""
		maxsm = 0
		for pword in proposed:
			if type(pword) is tuple:
				pword = list(pword)
			csm = -1
			if pword in k:
				csm = sm[k.index(pword)]
			result.append(csm)
			if csm > maxsm:
				maxsm = csm
				maxword = pword
#			print('%s %s' % (pword, csm))
		return result, maxword, maxsm

	def critique(self, sentence):
		tokens = sentence.split()
		tasks = split_sentence_for_eval(tokens, self._keyword_list, self._num_before, self._num_after)
		corrections = []
		for task in tasks:
			if task[1] == 0:
				continue
#			print(task)
			kw = self._keyword_list[task[1]]
			if type(kw) is list:
				kw = tuple(kw)
			proposed = self._keyword_map[kw]
			result, maxword, maxsm = self.lookup_prob(task[0], proposed)
			target_word = maxword
#			if type(target_word) is not list:
#				target_word = [target_word]
			src_word = tokens[task[2]:task[3]]
			if src_word != target_word and target_word not in src_word:
				corrections.append([task[2], task[3], src_word, target_word])
		return corrections, tokens

	def markup_critique(self, corrections, tokens):
		# TODO: move this out to a base class
		result = ""
		i = 0
		cur_corrections = copy.deepcopy(corrections)
		while (i < len(tokens)): # can escape early  if we run out of critiques
			if len(result)>0:
				result += ' '
			if len(cur_corrections)>0 and i >= cur_corrections[0][0]:
				i = cur_corrections[0][1]
				src_word = cur_corrections[0][2]
				if type(cur_corrections[0][3]) is not tuple and type(cur_corrections[0][3]) is not list:
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
	e = CLMEval()
	e.load("./output/clmV0/")
	sentence = "keep playing and you will loose"
	sentence = "if you keep playing , then you will loose"
	corrections, tokens = e.critique(sentence)
	markup = e.markup_critique(corrections, tokens)
	print(markup)

