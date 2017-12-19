import os


model_name = 'determinerCCNNV1'
params = { 'model_name': model_name,
					 'output_location': './output/%s/'%model_name,
					 'null_sample_factor': -1,  # <0= equal null as non null per sentence, 0 = don't do anything, >0 = factor
					 'num_words_before': 5,
					 'num_words_after': 5,
					 'embedding_size': 512,
					 'word_len': 5,
					 'use_char_cnn': True,
					 'conv_num_features': [[512, 512]],
					 'conv_widths': [[2,2]],
					 'conv_keep_probs': None,
					 'use_no_conv_path': True,                       # enable embedding pass through to second stage
					 'max_vocab_size': 20000,
					 'mlp_config': [512,128],
					 'bipass_conv': False,
					 'mlp_activations': 'sigmoid',
					 'mlp_keep_probs': 0.9,
					 'embedding_device': '/cpu:0',
					 'batch_size': 8192,
					 'learning_rate': 0.001,
					 'keywords': ['a','an','the'],
					 'mini_batches_between_checkpoint': 100,
					 'embedding_keep_prob': 0.9,
					 'start_token': "<S>",
					 'embedding_wd': 0.0001,                           # L2 WD regularization constant
					 'enable_regularization': True,
					 'monolingual_dir': '/mnt/work/1-billion-word-language-modeling-benchmark',
					 'vocab_file': '/mnt/work/1-billion-word-language-modeling-benchmark/1b_word_vocab.txt'
					 }
