from framework.utils.data.text_indexer import TextIndexer
from framework.evaluator import Evaluator
import framework.utils.common as utils
import os
import numpy as np
run_server = False

params = utils.load_param_file('params.py')

vocab_file = os.path.join(utils.get_dict_value(params,'output_location'), 'vocab.pkl')
ckpt = os.path.join(utils.get_dict_value(params,'output_location'),
										utils.get_dict_value(params, 'model_name') + '.ckpt')

sentences = ['The apple , which is rotten is not edible']

e = Evaluator.load2(ckpt)
i = TextIndexer.from_file(vocab_file)

num_before = utils.get_dict_value(params, "num_words_before")
num_after = utils.get_dict_value(params, "num_words_after")
pad_tok = utils.get_dict_value(params, "pad_tok", '<pad>')

sentence = "In simple terms , high precision means that an algorithm " \
			"returned substantially more relevant results than irrelevant ones , while" \
			" high recall means that an algorithm returned most of the relevant results ."

sentence = "<S> In simple terms , high precision means that algorithm " \
			"returned substantially more relevant results than irrelevant ones , while" \
			" high recall means that algorithm returned most of relevant results ."
#sentence = "<S> Precision can be seen as measure of exactness or quality , "\
#	"whereas recall is measure of completeness or quantity"
#sentence = "<S> I have to take kids to school today because teacher was not happy ."
#sentence = "Information and discussion by Microsoft and community"

sentence = ['<pad>']*10 + sentence.split() + ['<pad>']*10
print(sentence)
for toki, tok in enumerate(sentence):
	if (toki > len(sentence)-20):
		break
	csentence = sentence[toki:(toki+20)]
	_,indexed,_,_ = i.index_wordlist(csentence)
	r = e.eval({'sentence': [indexed]}, {'sm_decision'})
	rr = r[0][0]
	print("%s: %s %s"%(' '.join(["%0.4f"%x for x in rr]),np.argmax(r),csentence[10]))
