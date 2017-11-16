import os


model_name = 'thatthanV1'
params = { 'model_name': model_name,
					 'output_location': './output/%s/'%model_name,
					 'null_sample_factor': 0,  # <0= equal null as non null per sentence, 0 = don't do anything, >0 = factor
					 'num_words_before': 20,
					 'num_words_after': 20,
					 'embedding_size': 500,
					 'conv_num_features': [[300, 300, 300, 300, 300]],
					 'conv_widths': [[3, 3, 3, 3, 3]],
					 'add_redundant_keyword_data': False,
					 'ignore_negative_data': True,
					 'conv_keep_probs': None,
					 'use_no_conv_path': True,                       # enable embedding pass through to second stage
					 'mlp_config': [512],
					 'bipass_conv': False,
					 'mlp_activations': 'sigmoid',
					 'mlp_keep_probs': 0.9,
					 'embedding_device': '/cpu:0',
					 'batch_size': 8192,
					 'learning_rate': 0.0005,
					 'keywords': ['that','than'],
					 'mini_batches_between_checkpoint': 100,
					 'start_token': "<S>",
					 'embedding_wd': 0.0001,                           # L2 WD regularization constant
					 'enable_regularization': True,
					 'monolingual_dir': '/mnt/work/1-billion-word-language-modeling-benchmark',
					 'vocab_file': '/mnt/work/1-billion-word-language-modeling-benchmark/1b_word_vocab.txt'
					 }
