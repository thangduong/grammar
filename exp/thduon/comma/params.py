import os


model_name = 'commaV12'
params = { 'model_name': model_name,
					 'output_location': './output/%s/'%model_name,
					 'null_sample_factor': 0,  # <0= equal null as non null per sentence, 0 = don't do anything, >0 = factor
					 'num_words_before': 5,
					 'num_words_after': 5,
					 'embedding_size': 16,
					 'conv_num_features': [[128, 64]],
					 'conv_widths': [[2,2]],
					 'conv_keep_probs': None,
					 'use_no_conv_path': False,                       # enable embedding pass through to second stage
					 'mlp_config': [32],
					 'bipass_conv': False,
#					 'max_vocab_size': 20000,
					 'min_vocab_freq': 100,
					 'all_lowercase': True,
					 'mlp_activations': 'sigmoid',
					 'mlp_keep_probs': 0.9,
					 'embedding_device': '/cpu:0',
					 'batch_size': 1024*32,
					 'embedding_keep_prob': .9,
					 'learning_rate': 0.0001,
					 'keywords': [','],
					 'mini_batches_between_checkpoint': 100,
					 'embedding_wd': 0.0001,                           # L2 WD regularization constant
					 'start_token': '<S>',
					 'enable_regularization': True,
					 'monolingual_dir': '/mnt/work/1-billion-word-language-modeling-benchmark',
					 'vocab_file': '/mnt/work/1-billion-word-language-modeling-benchmark/lc_vocab.txt'
					 }
