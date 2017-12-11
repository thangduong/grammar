import os
import framework.subgraph.losses as losses

model_name = 'clmV4'
params = { 'model_name': model_name,
					 'output_location': './output/%s/'%model_name,
					 'null_sample_factor': 0,  # <0= equal null as non null per sentence, 0 = don't do anything, >0 = factor
					 'num_words_before': 10,
					 'num_words_after': 10,
					 'embedding_size': 150,
					 'conv_num_features': [[500, 400, 300]],
					 'conv_widths': [[2, 2, 2]],
					 'conv_keep_probs': None,
					 'all_lowercase': True,
					 'use_no_conv_path': True,                       # enable embedding pass through to second stage
					 'mlp_config': [256],
					 'bipass_conv': False,
					 'mlp_activations': 'sigmoid',
					 'mlp_keep_probs': 0.9,
					 'embedding_device': '/cpu:0',
					 'batch_size': 1024*8,
					 'learning_rate': 0.001,
					 'data_sampling': 'freeform',
					 'loss_function': losses.sampled_softmax_xentropy,
					 'ignore_negative_data': True,
					 'start_token': '<s>',
					 'unk_token': 'unk',
					 'keywords': [],
					 'mini_batches_between_checkpoint': 100,
					 'embedding_wd': 0.0001,                           # L2 WD regularization constant
					 'enable_regularization': True,
						'keywords_as_map': True,
					 'monolingual_dir': '/mnt/work/1-billion-word-language-modeling-benchmark',
					 'vocab_file': '../data/filtered_lowercase_vocab.txt'
					 }
