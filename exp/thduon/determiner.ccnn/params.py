import os


model_name = 'determinerCCNNV51'
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
					 'use_no_conv_path': False,                       # enable embedding pass through to second stage
					 'min_vocab_freq': 10,
					 'all_lowercase': True,
					 'mlp_config': [300,100],
					 'bipass_conv': False,
					 'mlp_activations': 'sigmoid',
					 'embedding_device': '/cpu:0',
					 'batch_size': 8192,
					 'learning_rate': 0.001,
					 'keywords': ['a','an','the'],
					 'mini_batches_between_checkpoint': 100,
					 'start_token': "<S>",
					 'conv_keep_probs': 0.75,
					 'char_conv_keep_probs': 0.75,
					 'mlp_keep_probs': 0.5,
					 'embedding_keep_prob': 0.75,
					 'embedding_wd': 0.001,                           # L2 WD regularization constant
					 'char_weight_wd_regularization': 0.001,                           # L2 WD regularization constant
					 'char_bias_wd_regularization': 0.001,                           # L2 WD regularization constant
					 'word_weight_wd_regularization': 0.001,                           # L2 WD regularization constant
					 'word_bias_wd_regularization': 0.001,                           # L2 WD regularization constant
					 'enable_regularization': True,
					 'training_data_dir': '/mnt/work/tokenized_training_data/',
					 'vocab_file': '../data/filtered_lowercase_vocab.txt'
					 }

"""
model_name = 'determinerCCNNV50'
params = { 'model_name': model_name,
					 'output_location': './output/%s/'%model_name,
					 'null_sample_factor': 0,  # <0= equal null as non null per sentence, 0 = don't do anything, >0 = factor
					 'num_words_before': 10,
					 'num_words_after': 10,
					 'embedding_size': 300,
					 'word_embedding_size': 100,
					 'word_len': 15,
					 'use_char_cnn': True,
					 'ccnn_num_words': 5,
					 'ccnn_skip_nonalphanumeric': True,
					 'char_conv_num_features': [[200, 100, 50]],
					 'char_conv_widths': [[3,3,3]],
					 'conv_num_features': [[1000,500,300,300,300]],
					 'conv_widths': [[9,3,3,3,3]],
					 'char_use_no_conv_path': True,
					 'use_no_conv_path': True,                       # enable embedding pass through to second stage
					 'all_lowercase': False,
					 'lowercase_char_path': False,
					 'mlp_config': [300, 100],
					 'bipass_conv': False,
					 'mlp_activations': 'sigmoid',
					 'batch_size': 8192,
					 'learning_rate': 0.0001,
					 'keywords': ['a','an','the', 'A', 'An', 'The'],
					 'start_sentence_synthetic_capital_and_lowercase': True,
					 'mini_batches_between_checkpoint': 100,
#					 'embedding_device': '/cpu:0',
#					 'min_vocab_freq': 50,
#					 'min_vocab_freq': 10,
					 'start_token': "<s>",
					 'conv_keep_probs': 0.75,
					 'char_conv_keep_probs': 0.75,
					 'mlp_keep_probs': 0.5,
					 'embedding_keep_prob': 0.75,
					 'embedding_wd': 0.001,                           # L2 WD regularization constant
					 'char_weight_wd_regularization': 0.001,                           # L2 WD regularization constant
					 'char_bias_wd_regularization': 0.001,                           # L2 WD regularization constant
					 'word_weight_wd_regularization': 0.001,                           # L2 WD regularization constant
					 'word_bias_wd_regularization': 0.001,                           # L2 WD regularization constant
					 'enable_regularization': True,
					 'training_data_dir': '/mnt/work/training_data.tok4',
					 'vocab_file': '/mnt/work/training_data.tok4/vocab/vocab.50.txt',
					 }
"""