import os


model_name = 'clcV0'
params = { 'model_name': model_name,
					 'output_location': './output/%s/'%model_name,
					 'null_sample_factor': 0,  # <0= equal null as non null per sentence, 0 = don't do anything, >0 = factor
					 'num_before': 50,
					 'num_after': 50,
					 'embedding_size': 200,
					 'conv_num_features': [[200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200]],
					 'conv_widths': [[2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]],
					 'conv_keep_probs': None,
					 'use_no_conv_path': False,                       # enable embedding pass through to second stage
					 'mlp_config': [512, 128],
					 'bipass_conv': False,
					 'mlp_activations': 'sigmoid',
					 'embedding_device': '/cpu:0',
					 'batch_size': 1024*16,
					 'learning_rate': 0.001,
					 'keywords': [','],
					 'mini_batches_between_checkpoint': 100,
					 'mlp_keep_probs': 0.5,
					 'conv_keep_probs': 0.75,
					 'weight_wd_regularization': 0.001,
					 'bias_wd_regularization': 0.001,
					 'embedding_wd': 0.001,                           # L2 WD regularization constant
					 'enable_regularization': True,
					 'training_data_dir': '/mnt/work/training-monolingual/',
					 }
