import os


model_name = 'clmV0'
params = { 'model_name': model_name,
					 'output_location': './output/%s/'%model_name,
					 'null_sample_factor': 0,  # <0= equal null as non null per sentence, 0 = don't do anything, >0 = factor
					 'num_words_before': 10,
					 'num_words_after': 10,
					 'embedding_size': 500,
					 'conv_num_features': [[500, 500, 500]],
					 'conv_widths': [[2, 2, 2]],
					 'conv_keep_probs': None,
					 'use_no_conv_path': True,                       # enable embedding pass through to second stage
					 'mlp_config': [512],
					 'bipass_conv': False,
					 'mlp_activations': 'sigmoid',
					 'mlp_keep_probs': 0.9,
					 'embedding_device': '/cpu:0',
					 'batch_size': 8192,
					 'learning_rate': 0.001,
					 'start_token': '<S>',
					 'unk_token': 'unk',
					 'keyword_buckets': [
						 [('who', "'s"), 'whose'],
						 ['than', 'then'],
						 ['their', 'there', ('they', "'re")],
						 ['accept', 'except'],
						 ['board', 'bored'],
						 ['advice', 'advise'],
						 ['affect', 'effect'],
						 ['among', 'amonst', 'between'],
						 ['assure', 'ensure', 'insure'],
						 ['breath', 'breathe'],
						 ['capital', 'capitol'],
						 ['complement', 'compliment'],
						 ['disinterested', 'uninterested'],
						 ['defence', 'defense'],
						 ['emigrate', 'immigrate'],
						 ['i.e.', 'e.g.'],
						 ['empathy', 'sympathy'],
						 ['farther', 'further'],
						 ['flaunt', 'flout'],
						 ['gaffe', 'gaff'],
						 ['gray', 'grey'],
						 ['historic', 'historical'],
						 ['imply', 'infer'],
						 [('it', "'s"), 'its'],
						 ['lay', 'lie'],
						 ['lead', 'led'],
						 ['learned', 'learnt'],
						 ['loose', 'lose'],
						 ['principal', 'principle'],
						 ['stationary', 'stationery'],
						 ['inquiry', 'enquiry'],
						 ['elicit', 'illicit'],
						 ['too', 'to', 'two'],
						 ['pear', 'pair', 'pare'],
						 ['tea', 'tee'],
						 ['poll', 'pole'],
						 ['male', 'mail'],
						 ['holy', 'wholly', 'holey', 'holly'],
						 ['half', 'have'],
						 ['altar', 'alter'],
						 ['ad', 'add', 'ads', 'adds'],
						 ['hall', 'haul'],
						 ['steal', 'steel'],
						 ['lets', ('let', "'s")],
						 ['bear', 'bare'],
						 ['plain', 'plane'],
						 ['for', 'four', 'fore', 'forth', 'fourth'],
						 ['may'], ['march']
					 ],
					 'keywords': [],
					 'mini_batches_between_checkpoint': 100,
					 'embedding_wd': 0.0001,                           # L2 WD regularization constant
					 'enable_regularization': True,
					 'monolingual_dir': '/mnt/work/1-billion-word-language-modeling-benchmark',
					 'vocab_file': '/mnt/work/1-billion-word-language-modeling-benchmark/1b_word_vocab.txt'
					 }
"""
'than','then','their','there', \
												['they', "'re"], 'pair','pear',
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
												'a','an','the', \
												'unk'
"""