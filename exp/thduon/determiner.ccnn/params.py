import os


model_name = 'determinerCCNNV15'
params = { 'model_name': model_name,
					 'output_location': './output/%s/'%model_name,
					 'null_sample_factor': 0,  # <0= equal null as non null per sentence, 0 = don't do anything, >0 = factor
					 'num_words_before': 10,
					 'num_words_after': 10,
					 'embedding_size': 300,
					 'word_len': 15,
					 'use_char_cnn': True,
					 'ccnn_num_words': 3,
					 'char_use_no_conv_path': True,
					 'char_conv_num_features': [[50, 50]],
					 'char_conv_widths': [[2, 2]],
					 'conv_num_features': [[300,300,300,300,300,300]],
					 'conv_widths': [[2,2,2,2,2,2]],
					 'conv_keep_probs': None,
					 'use_no_conv_path': False,                       # enable embedding pass through to second stage
					 'min_vocab_freq': 10,
					 'all_lowercase': True,
					 'lowercase_char_path': False,
					 'mlp_config': [300, 100],
					 'bipass_conv': False,
					 'mlp_activations': 'sigmoid',
					 'mlp_keep_probs': 0.9,
#					 'embedding_device': '/cpu:0',
					 'batch_size': 8192,
					 'learning_rate': 0.001,
					 'keywords': ['a','an','the'],
					 'mini_batches_between_checkpoint': 100,
					 'embedding_keep_prob': 0.9,
#					 'min_vocab_freq': 50,
					 'start_token': "<s>",
					 'embedding_wd': 0.0001,                           # L2 WD regularization constant
					 'enable_regularization': True,
					 'training_data_dir': '/mnt/work/training_data.tok2',
					 'vocab_file': '/mnt/work/training_data.tok2/vocab/filtered_lowercase_vocab.txt',
					 }
