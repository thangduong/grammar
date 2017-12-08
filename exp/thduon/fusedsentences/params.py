import os


model_name = 'fusedsentencesV3'
params = { 'model_name': model_name,
					 'output_location': './output/%s/'%model_name,
					 'null_sample_factor': 0,  # <0= equal null as non null per sentence, 0 = don't do anything, >0 = factor
					 'num_words_before': 40,
					 'num_words_after': 40,
					 'embedding_size': 300,
					 'conv_num_features': [[300,300,300,300,300,300,300]],
					 'conv_widths': [[2,2,2,2,2,2,2]],
					 'all_lowercase': True,
					 'conv_keep_probs': None,
					 'use_no_conv_path': False,                       # enable embedding pass through to second stage
					 'mlp_config': [512],
					 'bipass_conv': False,
					 'mlp_activations': 'sigmoid',
					 'mlp_keep_probs': 0.9,
					 'embedding_device': '/cpu:0',
					 'batch_size': 4*1024,
					 'learning_rate': 0.001,
					 'keywords': [','],
					 'mini_batches_between_checkpoint': 100,
					 'embedding_wd': 0.0001,                           # L2 WD regularization constant
					 'start_token': '<s>',
					 'enable_regularization': True,
					 'monolingual_dir': '/mnt/work/1-billion-word-language-modeling-benchmark',
					 'vocab_file': '/mnt/work/1-billion-word-language-modeling-benchmark/vocab_small_lc.txt'
					 }
