from framework.utils.data.text_indexer import TextIndexer
from framework.evaluator import Evaluator
import framework.utils.common as utils
import os

params = utils.load_param_file('params.py')

vocab_file = os.path.join(utils.get_dict_value(params,'output_location'), 'vocab.pkl')
ckpt = os.path.join(utils.get_dict_value(params,'output_location'),
										utils.get_dict_value(params, 'model_name') + '.ckpt')

sentences = ['Jimenez reported from San Jose Costa Rica ; Associated Press',
             "We went to the store and I bought a banana",
						 "This is the cat that walked over the mat .",
						 "<pad> <pad> In this case I think the problem is"]

e = Evaluator.load2(ckpt)
i = TextIndexer.from_file(vocab_file)

for sentence in sentences:
	for j in range(5):
		sentence = '<pad> ' + sentence + " <pad>"
	_,indexed,_,_ = i.index_wordlist(sentence.split())
	r = e.eval({'sentence': [indexed]}, {'sm_decision'})
	print(sentence)
	print(r[0][0][1])