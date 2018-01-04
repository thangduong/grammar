import os
import framework.subgraph.losses as losses

model_name = 'clmV7'
params = { 'model_name': model_name,
					 'output_location': './output/%s/'%model_name,
					 'null_sample_factor': 0,  # <0= equal null as non null per sentence, 0 = don't do anything, >0 = factor
					 'num_words_before': 10,
					 'num_words_after': 10,
					 'embedding_size': 150,
					 'conv_num_features': [[200]],
					 'conv_widths': [[2]],
					 'conv_keep_probs': None,
					 'all_lowercase': True,
					 'use_no_conv_path': True,                       # enable embedding pass through to second stage
					 'mlp_config': [256],
					 'bipass_conv': False,
					 'mlp_activations': 'sigmoid',
					 'mlp_keep_probs': 0.9,
					 'embedding_device': '/cpu:0',
					 'batch_size': 1024,
					 'learning_rate': 0.001,
					 'data_sampling': 'freeform',
#					 'max_vocab_size': 50000,
#					 'min_vocab_freq': 1000,
					 'loss_function': losses.softmax_xentropy,
					 'ignore_negative_data': True,
					 'start_token': '<s>',
					 'unk_token': 'unk',
					 'keywords': [],
					 'mini_batches_between_checkpoint': 100,
					 'embedding_wd': 0.0001,                           # L2 WD regularization constant
					 'enable_regularization': True,
						'keywords_as_map': True,
					 'training_data_dir': '/mnt/work/training_data.tok2/',
					 'vocab_file': '/mnt/work/training_data.tok2/vocab/merged_vocab_500.txt',
					 'out_vocab_file': '',
					 }
