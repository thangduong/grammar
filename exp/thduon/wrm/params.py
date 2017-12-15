import os
import copy

model_name = 'wrmV16'
params = { 'model_name': model_name,
					 'output_location': './output/%s/'%model_name,
					 'null_sample_factor': 0,  # <0= equal null as non null per sentence, 0 = don't do anything, >0 = factor
					 'num_words_before': 10,
					 'num_words_after': 10,
					 'embedding_size': 300,
					 'conv_num_features': [[200, 200, 200]],
					 'conv_widths': [[2,2,2]],
					 'conv_keep_probs': None,
					 'all_lowercase': True,
					 'use_no_conv_path': True,                       # enable embedding pass through to second stage
					 'mlp_config': [256],
					 'bipass_conv': False,
					 'mlp_activations': 'sigmoid',
					 'mlp_keep_probs': 0.9,
					 'embedding_keep_prob': 0.8,
					 'embedding_device': '/cpu:0',
					 'batch_size': 8192,
					 'learning_rate': 0.0005,
					 'start_token': '<S>',
					 'unk_token': 'unk',
#					 'max_vocab_size': 50000,
					 'keywords': {},
					 'keyword_buckets': [
						 ['their','there',('they',"'re")]
					 ],
					 'mini_batches_between_checkpoint': 100,
					 'embedding_wd': 0.0001,                           # L2 WD regularization constant
					 'enable_regularization': True,
					 'training_data_dir': '/mnt/work/tokenized_wiki_training/',
					 'vocab_file': 'vocab_lcase.txt'
					 }

def gen_keywords(params):
	buckets = params['keyword_buckets']
	keywords = params['keywords']
	all_lowercase = params['all_lowercase']
	idx = 0
	repeat_count = 0
	id_to_keyword = []
	for bucket in buckets:
		words_to_add = []
		if not all_lowercase:
			for keyword in bucket:
				if type(keyword) is tuple:
					words_to_add.append(tuple(' '.join(list(keyword)).capitalize().split()))
				else:
					words_to_add.append(keyword.capitalize())
			print("GENERATED UPPER CASE VERSION")
		bucket += words_to_add
		for keyword in bucket:
			if keyword not in keywords:
				keywords[keyword] = [idx]+copy.deepcopy(bucket)
				keywords[keyword].remove(keyword)
				id_to_keyword.append(keyword)
				idx += 1
			else:
				print("REPEATED %s" % keyword)
				repeat_count += 1
				cur_idx = keywords[keyword][0]
				newlist = set()
				newlist = newlist.union(keywords[keyword][1:])
				newlist = newlist.union(bucket)
				keywords[keyword] = [cur_idx] + list(newlist)
				keywords[keyword].remove(keyword)
	params['id_to_keyword'] = id_to_keyword
	if repeat_count > 0:
		exit(0)
	print(id_to_keyword)
	return params

gen_keywords(params)
