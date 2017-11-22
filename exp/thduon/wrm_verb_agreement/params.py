import os
import copy

model_name = 'wrmV6'
params = { 'model_name': model_name,
					 'output_location': './output/%s/'%model_name,
					 'null_sample_factor': 0,  # <0= equal null as non null per sentence, 0 = don't do anything, >0 = factor
					 'num_words_before': 10,
					 'num_words_after': 10,
					 'embedding_size': 100,
					 'conv_num_features': [[200, 200, 200, 200, 200]],
					 'conv_widths': [[3,3,3,3,3]],
					 'conv_keep_probs': None,
					 'use_no_conv_path': True,                       # enable embedding pass through to second stage
					 'mlp_config': [512],
					 'bipass_conv': False,
					 'mlp_activations': 'sigmoid',
					 'mlp_keep_probs': 0.9,
					 'embedding_keep_prob': 0.8,
					 'embedding_device': '/cpu:0',
					 'batch_size': 8192,
					 'learning_rate': 0.0005,
					 'start_token': '<S>',
					 'unk_token': 'unk',
					 'max_vocab_size': 10000,
					 'keywords': {},
					 'keyword_buckets': [
						 ['come','comes','came'],
						 ['chase','chases','chased'],
						 ['use','used','uses'],
						 ['find','finds','found'],
						 ['is','are','was','were'],
						 ['do','does','did'],
						 ['say','said','says'],
						 ['go','goes','went'],
						 ['make','makes','made'],
						 ['take','takes','took','taked'],
						 ['see','sees','saw','seed'],
						 ['gives','give','gave']
					 ],
					 'mini_batches_between_checkpoint': 100,
					 'embedding_wd': 0.0001,                           # L2 WD regularization constant
					 'enable_regularization': True,
					 'monolingual_dir': '/mnt/work/1-billion-word-language-modeling-benchmark',
					 'vocab_file': '/mnt/work/1-billion-word-language-modeling-benchmark/1b_word_vocab.txt'
					 }


def gen_keywords(params):
	buckets = params['keyword_buckets']
	keywords = params['keywords']
	idx = 0
	id_to_keyword = []
	for bucket in buckets:
		words_to_add = []
		for keyword in bucket:
			if type(keyword) is tuple:
				words_to_add.append(tuple(' '.join(list(keyword)).capitalize().split()))
			else:
				words_to_add.append(keyword.capitalize())
		bucket += words_to_add
		for keyword in bucket:
			keywords[keyword] = [idx]+copy.deepcopy(bucket)
			keywords[keyword].remove(keyword)
			id_to_keyword.append(keyword)
			idx += 1
	params['id_to_keyword'] = id_to_keyword
	print(id_to_keyword)
	return params


gen_keywords(params)
