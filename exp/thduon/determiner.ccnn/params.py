import os


model_name = 'determinerCCNNV37'
params = { 'model_name': model_name,
					 'output_location': './output/%s/'%model_name,
					 'null_sample_factor': 0,  # <0= equal null as non null per sentence, 0 = don't do anything, >0 = factor
					 'num_words_before': 10,
					 'num_words_after': 10,
					 'embedding_size': 300,
					 'word_len': 15,
					 'use_char_cnn': True,
					 'ccnn_num_words': 3,
					 'conv_num_features': [[300,300,300,300,300,300]],
					 'conv_widths': [[2,2,2,2,2,2]],
					 'conv_keep_probs': None,
					 'use_no_conv_path': False,                       # enable embedding pass through to second stage
					 'min_vocab_freq': 10,
					 'all_lowercase': True,
					 'mlp_config': [300,100],
					 'bipass_conv': False,
					 'mlp_activations': 'sigmoid',
					 'mlp_keep_probs': 0.9,
					 'embedding_device': '/cpu:0',
					 'batch_size': 1024*4,
					 'learning_rate': 0.001,
					 'keywords': ['a','an','the'],
					 'mini_batches_between_checkpoint': 100,
					 'embedding_keep_prob': 0.9,
					 'start_token': "<S>",
					 'embedding_wd': 0.0001,                           # L2 WD regularization constant
					 'enable_regularization': True,
					 'training_data_dir': '/mnt/work/tokenized_training_data/',
					 'vocab_file': '../data/filtered_lowercase_vocab.txt'
					 }
