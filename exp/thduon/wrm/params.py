import os
import copy

model_name = 'wrmV8'
params = { 'model_name': model_name,
					 'output_location': './output/%s/'%model_name,
					 'null_sample_factor': 0,  # <0= equal null as non null per sentence, 0 = don't do anything, >0 = factor
					 'num_words_before': 10,
					 'num_words_after': 10,
					 'embedding_size': 100,
					 'conv_num_features': [[200, 200, 200, 200, 200, 200]],
					 'conv_widths': [[2,2,2,2,2, 2]],
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
					 'max_vocab_size': 50000,
					 'keywords': {},
					 'keyword_buckets': [
						 [('who', "'s"), 'whose'],
						 ['than','then'],
						 ['their','there',('they',"'re")],
						 ['accept','except'],
						 ['board', 'bored'],
						 ['advice', 'advise'],
						 ['affect','effect'],
						 ['among','amonst','between'],
						 ['assure','ensure','insure'],
						 ['breath', 'breathe'],
						 ['capital','capitol'],
						 ['complement','compliment'],
						 ['disinterested', 'uninterested'],
						 ['defence', 'defense'],
						 ['emigrate','immigrate'],
						 ['i.e.','e.g.'],
						 ['empathy','sympathy'],
						 ['farther','further'],
						 ['flaunt','flout'],
						 ['gaffe', 'gaff'],
						 ['gray','grey'],
						 ['historic','historical'],
						 ['imply','infer'],
						 [('it',"'s"),'its','it'],
						 ['lay','lie'],
						 ['lead','led'],
						 ['learned','learnt'],
						 ['loose', 'lose'],
						 ['principal','principle'],
						 ['stationary','stationery'],
						 ['inquiry','enquiry'],
						 ['elicit', 'illicit'],
						 ['too','to','two'],
						 ['pear','pair', 'pare'],
						 ['tea','tee'],
						 ['poll','pole'],
						 ['male','mail'],
						 ['holy', 'wholly', 'holey', 'holly'],
						 ['half', 'have'],
						 ['altar','alter'],
						 ['ad','add', 'ads', 'adds'],
						 ['hall', 'haul'],
						 ['steal','steel'],
						 ['lets', ('let',"'s")],
						 ['bear','bare'],
						 ['plain','plane'],
						 ['for','four','fore','forth','fourth'],
						 ['may'], ['march'],
						 ['a','an','the'],
						 ['quiet','quite'],
						 ['sole','soul'],
						 ['soles','souls'],
						 ['your',('you',"'re")],
						 [('all','right'),'alright'],
						 [('some','time'),'sometime','sometimes'],
						 ['everyday',('every','day')],
						 ['hoard','whore'],
						 ['i'],
						 ['besides','beside'],
						 ['on','in'],
						 ['around','about']
						 ['from','of']
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
