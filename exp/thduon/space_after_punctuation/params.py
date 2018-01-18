import os

model_name = 'sapV0'
params = { 'model_name': model_name,
					 'output_location': './output/%s/'%model_name,
					 'num_chars_before': 15,
					 'num_chars_after': 15,
					 'lines_per_group': 10,
					 'vocab_size': 256,
					 'punctuations': [',','.',';',':','!','?','"', "'", ')'],
					 'training_data_dir': '/mnt/work/training-monolingual/',
					 'conv_num_features': [[500,500]],
					 'conv_widths': [[3,3]],
					 'mlp_config': [512,256],
					 'bipass_conv': False,
					 'mlp_activations': 'sigmoid',
					 'use_no_conv_path': False,
					 'embedding_size': 100,
					 'embedding_device': None,
					 'embedding_keep_prob': 0.5,
					 'conv_keep_probs': 0.5,
					 'mlp_keep_probs': 0.5,
					 'weight_wd_regularization': 0.001,
					 'bias_wd_regularization': 0.001,
					 'embedding_wd': 0.001,
					 'batch_size': 1024*8,
					 'enable_regularization': True,
					 'mini_batches_between_checkpoint': 100,
					 'num_classes': 2, #0=no space, 1=space
					 }
