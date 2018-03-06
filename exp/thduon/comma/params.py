import os


model_name = 'commaV51'
params = { 'model_name': model_name,
					 'output_location': './output/%s/'%model_name,
					 'null_sample_factor': 0,  # <0= equal null as non null per sentence, 0 = don't do anything, >0 = factor
					 'num_words_before': 15
					 'num_words_after': 15,
					 'embedding_size': 300,
					 'conv_num_features': [[300,300,300,300,300,300,300]],
					 'conv_widths': [[3,3,3,3,3,3,3]],
					 'use_no_conv_path': True,                       # enable embedding pass through to second stage
					 'mlp_config': [512,256],
					 'bipass_conv': False,
					 'mlp_activations': 'sigmoid',
					 'mlp_keep_probs': 0.9,
					 'embedding_device': '/cpu:0',
					 'batch_size': 8192,
					 'embedding_keep_prob': .8,
					 'learning_rate': 0.0001,
					 'all_lowercase': True,
					 'keywords': [','],
					 'mini_batches_between_checkpoint': 100,
					 'embedding_wd': 0.0001,                           # L2 WD regularization constant
					 'start_token': '<s>',
					 'enable_regularization': True,
					 'data_dir': '/mnt/work/training_data/wiki-statmt-enron-reddit.tokenized',
					 'vocab_file': '/mnt/work/training_data/wiki-statmt-enron-reddit.tokenized/train/vocab/lowercase_vocab.50.txt',
					 }
