import os

model_name = 'sapV13'
params = { 'model_name': model_name,
					 'output_location': './output/%s/'%model_name,
					 'num_chars_before': 10,
					 'num_chars_after': 10,
					 'lines_per_group': 5,
					 'preserve_line_start': True,			# true makes sure the start of every line shows up as the start of at least one data packet
					 																	# e.g. s1, s2, s3.  input = [s1 shuf[s2,s3]], [s2 shuf[s1,s2]], etc.
					 'vocab_size': 10000,
					 'punctuations': [',','.',';',':','!','?',')',']','”','ʼ'],
#					 'training_data_dir': '/mnt/work/training-monolingual_with_reddit/',
					 'training_data_dir': '/mnt/work/statmt_reddit_shuf/',
					 'conv_num_features': [],#[25,25]],
					 'conv_widths': [],#[3,3]],
					 'mlp_config': [64,32,16],
					 'bipass_conv': False,
					 'mlp_activations': 'sigmoid',
					 'case_mode': 'normal_and_lower',		# 'normal' (default), 'lower', 'normal_and_lower'
					 'use_no_conv_path': True,
					 'embedding_size': 32,
					 'embedding_device': None,
					 'embedding_keep_prob': 0.5,
					 'conv_keep_probs': 0.5,
					 'mlp_keep_probs': 0.5,
					 'learning_rate': 0.0005,
					 'weight_wd_regularization': 0.001,
					 'bias_wd_regularization': 0.001,
					 'embedding_wd': 0.001,
					 'batch_size': 1024*16,
					 'enable_regularization': True,
					 'mini_batches_between_checkpoint': 100,
					 'num_classes': 2, #0=no space, 1=space
					 'inference_output_node': 'sm_decision',
					 }
