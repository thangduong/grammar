from framework.evaluator import Evaluator
import framework.utils.common as utils
from time import time
import numpy as np
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

params = utils.load_param_file(sys.argv[1])

ckpt = os.path.join(utils.get_dict_value(params,'output_location'),
										utils.get_dict_value(params, 'model_name') + '.ckpt')
gd = os.path.join(utils.get_dict_value(params,'output_location'),'release',
										utils.get_dict_value(params, 'model_name') + '.graphdef')
print("USING: %s"%gd)
e = Evaluator.load_graphdef(gd)
#e = Evaluator.load2(ckpt)
#e.dump_variable_sizes()

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


thres = .9
if len(sys.argv)>3:
	thres = float(sys.argv[3])
x = generate_data_from_sentence(params, sentence)

sbatch = []
idx = []
for a,b in x:
	print(' '.join([str(y) for y in a]))
	sbatch.append(a)
#	print([chr(x) for x in a])
	idx.append(b)
#print(idx)

result = e.eval({'sentence':sbatch},['sm_decision'])
result = result[0]
sentence_result = sentence
adjust = 0
print("thres = %0.2f"%thres)
for i,r in enumerate(result):
	j = idx[i]
	print(r)
	if r[1] > thres:
		jj = j + adjust
		sentence_result = sentence_result[:jj+1] + ' ' + sentence_result[jj+1:]
		adjust += 1
	print("%s,%s"%(sentence[j:(j+2)],r[0]))

print(sentence)
print(sentence_result)
