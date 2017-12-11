import random
import math
import numpy as np
from time import time

def gen_data(dataobj, tokens, keywords,
						 num_before=5, num_after=5,
						 pad_tok="<pad>", null_sample_factor=0,
						 add_redundant_keyword_data=True,
						 use_negative_only_data=True,
						 ignore_negative_data=False,
						 add_keyword_removal_data=False):
	dataobj._mean = np.mean(dataobj._y_count[1:])
	dataobj._std = np.std(dataobj._y_count[1:])
	dataobj._max = np.max(dataobj._y_count[1:])
	dataobj._min = np.min(dataobj._y_count[1:])
	tokens = [pad_tok] * num_before + tokens + [pad_tok]*(num_after+5)
	class_offset = 1
	if ignore_negative_data:
		class_offset = 0

	replaced_list = []
	original = []
	for toki in range(num_before, len(tokens)-num_before-5):
#		tok0 = tokens[toki].lower()
#		tok1 = tuple([x.lower() for x in tokens[toki:toki+2]])
#		tok2 = tuple([x.lower() for x in tokens[toki:toki+3]], an)
		tok0 = tokens[toki]
		tok1 = tuple([x for x in tokens[toki:toki+2]])
		tok2 = tuple([x for x in tokens[toki:toki+3]])
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
			ki = bucket[0]
			bucketid = [keywords[x][0] for x in bucket[1:]]
			bucketfreq = [dataobj._y_count[x+class_offset] for x in bucketid]
			max_bucket_freq = np.min(bucketfreq)
			if dataobj._y_count[ki+class_offset] < min(dataobj._mean + 5 + dataobj._std * 1.5,max_bucket_freq+2):
				for bad_word in bucket[1:]:
					if type(bad_word) is tuple:
						bad_word = list(bad_word)
					else:
						bad_word = [bad_word]
					test_sentence = tokens[(toki - num_before):toki] + bad_word + tokens[(toki + sl):(toki + num_after+5)]
					test_sentence = test_sentence[:(num_before+num_after)]
					replaced_list.append((test_sentence,ki + class_offset))
				replaced_list.append((tokens[(toki-num_before):(toki+num_after)], 0))
		original.append((tokens[(toki-num_before):(toki+num_after)], 0))
#	random.shuffle(original)
	results = replaced_list # + original[:len(replaced_list)]
	return results

random.seed(time())
if __name__ == "__main__":
	from word_classifier.data import ClassifierData
	import framework.utils.common as utils
	param_file = 'params.py'
	params = utils.load_param_file(param_file)
	print(params['keywords'])
	s = "they 're like , it 's better than yours"
	s = "there is no place like home"
	a = gen_data(s.split(), params['keywords'])
	print(a)




