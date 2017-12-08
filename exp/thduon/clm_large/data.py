import random
import math
def gen_data(dataobj, tokens, keywords,
						 num_before=5, num_after=5,
						 pad_tok="<pad>", null_sample_factor=0,
						 add_redundant_keyword_data=True,
						 use_negative_only_data=True,
						 ignore_negative_data=False,
						 add_keyword_removal_data=False):

	tokens = [pad_tok] * num_before + tokens + [pad_tok]*(num_after+5)
	class_offset = 1
	if ignore_negative_data:
		class_offset = 0
	results = []
	unk_list = []
	no_insert_list = []
	for toki in range(num_before, len(tokens)-num_before-4):
		tok0 = tokens[toki]
		if tok0 in keywords:
			ki = keywords[tok0]
			results.append( \
				(tokens[(toki - num_before):toki] + tokens[(toki + 1):(toki + num_after + 1)], \
				 ki + class_offset))
#		else:
			# add unk
#			if 'unk' in keywords:
#				ki = keywords.index('unk')
#				unk_list.append((tokens[(toki-num_before):toki]+tokens[(toki+1):(toki+num_after+1)], ki + class_offset))
#		no_insert_list.append((tokens[(toki-num_before):toki]+tokens[(toki):(toki+num_after)], 0))
#	num_to_add = min([int(len(results)/len(keywords)),len(no_insert_list),len(unk_list)])
#	if num_to_add == 0 and len(results)>0:
#		num_to_add = 1
#	random.shuffle(no_insert_list)
#	random.shuffle(unk_list)
#	if num_to_add > 0:
#		results += no_insert_list[:num_to_add]
#		results += unk_list[:num_to_add]
	return results


if __name__ == "__main__":
	from word_classifier.data import ClassifierData
	import framework.utils.common as utils

	sentence = '<S> they \'re trying to build a bridge'
	keywords = ['than', 'then', 'their', 'there', \
							 ['they', "'re"], 'pair', 'pear',
							 'accept', 'except', 'effect', 'affect', ['a', 'lot'], 'alot',
							 'allusion', 'illusion', 'illicit', 'elicit', \
							 'bad', 'badly', 'awhile', ['a', 'while'], \
							 'breath', 'breathe', \
							 'cache', 'cash', \
							 'complement', 'compliment', \
							 'desert', 'dessert', \
							 'deer', 'dear', \
							 'hone', 'home', \
							 'its', ['it', "'s"], \
							 'lead', 'led', \
							 'lose', 'loose', \
							 'precede', 'proceed', \
							 'passed', 'past', \
							 'principal', 'principle', \
							 'sell', 'sale',
							 'site', 'sight', \
							 'stationary', 'stationery', \
							 'unk', 'a','an','the'
							 ]
	param_file = 'params.py'
	params = utils.load_param_file(param_file)
	params['num_classes'] = len(params['keywords']) + 1
	d = ClassifierData.get_monolingual_training(base_dir=params['monolingual_dir'],
																													params=params,
																													gen_data_fcn=gen_data)
	d.next_batch(10)



	"""
			tok1 = tuple([x.lower() for x in tokens[toki:toki+2]])
		tok2 = tuple([x.lower() for x in tokens[toki:toki+3]])
		if tok2 in keywords:
			ki = keywords[tok2]
			results.append(\
				(tokens[(toki-num_before):toki]+tokens[(toki+3):(toki+num_after+3)], \
				ki + class_offset))
		elif tok1 in keywords:
			ki = keywords[tok1]
			results.append(\
				(tokens[(toki-num_before):toki]+tokens[(toki+2):(toki+num_after+2)], \
				ki + class_offset))
		el"""