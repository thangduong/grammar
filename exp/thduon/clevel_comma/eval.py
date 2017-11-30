from framework.evaluator import Evaluator
from data import _gen_data
import framework.utils.common as utils
import os
import numpy as np
import sys
run_server = False
def _gen_data(sentence, num_before, num_after):
	global max_value
	slen = len(sentence)
	z = [ord(x) for x in sentence if ord(x)>255]
	if len(z)>0 and max(z) > max_value:
		max_value = max(z)
		print('max_value = %s'%max_value)
	sentence = [min(ord(x),255) for x in sentence]
	sentence = [0]*(num_before-1) + [1] + sentence + [2] + [0]*(num_after - 1)

	null_list = []
	keychars = [ord(',')]
	for i in range(num_before, num_before+slen):
			null_list.append(sentence[i-num_before:i] + sentence[i:i+num_after])

	result = [[x,0] for x in null_list]
	for x in result:
		yield x

params = utils.load_param_file('params.py')

vocab_file = os.path.join(utils.get_dict_value(params,'output_location'), 'vocab.pkl')
ckpt = os.path.join(utils.get_dict_value(params,'output_location'),
										utils.get_dict_value(params, 'model_name') + '.ckpt')

sentences = ['The apple which is rotten is not edible'
						 ,'The quick brown fox jumped over the lazy dog and it ate the cat'
						 ,"Over a decade later it's the only top ten site run by a non-profit and a community of volunteers"
						 ,"In late August 1961 (only a few weeks after he was born) Barack and his mother moved to the University of Washington in Seattle where they lived for a year"
						 ,"Obama Sr. returned to Kenya in 1964 where he married for a third time."
#						 ,'Recalling his early childhood, Obama said "That my father looked nothing like the people around me -- that he was black as pitch my mother white as milk -- barely registered in my mind."'
#						 ,"From age six to ten Obama attended local Indonesian-language schools"
						 ]
sentences = [sys.argv[1]]
e = Evaluator.load2(ckpt)

num_before = utils.get_dict_value(params, "num_before")
num_after = utils.get_dict_value(params, "num_after")

for sentence in sentences:
	test = list(_gen_data(sentence, num_before, num_after))
	u = [chr(x) for x in test[0][0]]
#	print(u)

	idata = [x[0] for x in test]
	r = e.eval({'sentence': idata}, {'sm_decision'})
	r = r[0]

	#print(r)
	#print(len(sentence))
	#print(len(r))
	for x,y in zip(r, sentence):
		print('%s - %0.4f'%(y,x[1]))

	print("-------------------")
