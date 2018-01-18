from framework.evaluator import Evaluator
import framework.utils.common as utils
from time import time
import numpy as np
import os
import sys

params = utils.load_param_file(sys.argv[1])

vocab_file = os.path.join(utils.get_dict_value(params,'output_location'), 'vocab.pkl')
ckpt = os.path.join(utils.get_dict_value(params,'output_location'),
										utils.get_dict_value(params, 'model_name') + '.ckpt')

#e = Evaluator.load_graphdef('commaV10.graphdef')
e = Evaluator.load2(ckpt)

sentence = sys.argv[2]
def generate_data_from_sentence(params, sentence):
	punctuations = params['punctuations']
	num_chars_before = params['num_chars_before']
	num_chars_after = params['num_chars_after']
	vocab_size = params['vocab_size']
	ord_sentence = [0]*num_chars_before + [min(ord(x), vocab_size-1) for x in sentence] + [0]*num_chars_after

	for i in range(num_chars_before, num_chars_before + len(sentence) - 2):
		if chr(ord_sentence[i]) in punctuations:
			before = ord_sentence[(i - num_chars_before + 1):(i+1)]
			after = ord_sentence[(i+1):(i+num_chars_after+1)]
			yield before + after, i-num_chars_before


x = generate_data_from_sentence(params, sentence)

sbatch = []
idx = []
for a,b in x:
	sbatch.append(a)
#	print([chr(x) for x in a])
	idx.append(b)
#print(idx)

result = e.eval({'sentence':sbatch},['sm_decision'])
result = result[0]
print(sentence)
for i,r in enumerate(result):
	j = idx[i]
	print("%s,%s"%(sentence[j:(j+2)],r[1]))