import os


model_name = 'hyphenV0'
params = { 'model_name': model_name,
					 'output_location': './output/%s/'%model_name,
					 'null_sample_factor': 0,  # <0= equal null as non null per sentence, 0 = don't do anything, >0 = factor
					 'num_words_before': 5,
					 'num_words_after': 5,
					 'embedding_size': 100,
					 'conv_num_features': [[128, 64]],
					 'conv_widths': [[3,3]],
					 'use_no_conv_path': False,                       # enable embedding pass through to second stage
					 'mlp_config': [64, 32],
					 'bipass_conv': False,
#					 'max_vocab_size': 20000,
#					 'min_vocab_freq': 100,
					 'mlp_activations': 'sigmoid',
#					 'embedding_device': '/cpu:0',
					 'batch_size': 1024*8,
					 'learning_rate': 0.001,
					 'keywords': [','],
					 'mini_batches_between_checkpoint': 100,
					 'mlp_keep_probs': 0.75,
					 'conv_keep_probs': 0.75,
  				     'embedding_keep_prob': 0.75,
				     'weight_wd_regularization': 0.001,
				     'bias_wd_regularization': 0.001,
				     'embedding_wd': 0.001,                           # L2 WD regularization constant
					 'start_token': '<s>',
					 'enable_regularization': True,
					 'all_lowercase': True,
					 'training_data_dir': '/mnt/work/training_data.tok4/',
					 'vocab_file': '/mnt/work/training_data.tok4/vocab/lowercase_vocab.50.txt'
					 }
