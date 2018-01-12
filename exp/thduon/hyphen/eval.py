from framework.utils.data.text_indexer import TextIndexer
from framework.evaluator import Evaluator
import framework.utils.common as utils
import os
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '2'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys

paramsfile = "output/commaV5/params.py"

def generate_model_input_sentences(tokens, params):
	pad_tok = '<pad>'
	num_before = params['num_words_before']
	num_after = params['num_words_after']
	start_token = utils.get_dict_value(params, 'start_token')
	if start_token is not None and len(start_token) > 0:
		tokens = [pad_tok] * (num_before - 1) + [start_token] + tokens + [pad_tok] * (num_after + 5)
	else:
		tokens = [pad_tok] * num_before + tokens + [pad_tok] * (num_after + 5)
	result = []
	for toki in range(num_before, len(tokens) - num_before - 5):
		result.append(tokens[toki - num_before:toki] + tokens[toki:toki + num_after])
	return result


params = utils.load_param_file(paramsfile)

vocab_file = os.path.join(utils.get_dict_value(params,'output_location'), 'vocab.pkl')
ckpt = os.path.join(utils.get_dict_value(params,'output_location'),
										utils.get_dict_value(params, 'model_name') + '.ckpt')

gdfile = os.path.join(utils.get_dict_value(params,'output_location'),
		"release",
		utils.get_dict_value(params, 'model_name') + '.graphdef')

#e = Evaluator.load2(ckpt)
e = Evaluator.load_graphdef(gdfile)
i = TextIndexer.from_file(vocab_file)

sentence = "Linda owns a catering business in New Orleans She enjoys cooking for special events such as weddings , parties , and holidays "
sentence = "Driving home from school , Brett vowed to protect the fragile ecosystem all " \
			"the while the tires of his Cadillac Escalade flattened the toads hopping on the wet streets"
sentence = sys.argv[1]
tokens = sentence.lower().split()
mi = generate_model_input_sentences(tokens, params)
imi = []
for s in mi:
	a,indexed,b,c = i.index_wordlist(s)
	print(indexed)
	imi.append(indexed)
#	print(a)
#	print(b)
#	print(c)
r = e.eval({'sentence': imi}, {'sm_decision'})
for pr, tok in zip(r[0],tokens):
	print("%s - %0.4f"%(tok,pr[1]))
