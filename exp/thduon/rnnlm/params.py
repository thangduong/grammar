import os
import tensorflow as tf


model_name = 'rnnlmV51'
params = { 'model_name': model_name,
					 'output_location': './output/%s/'%model_name,
					 'training_data_dir': '/mnt/work/training_data.tok4/1b',
					 'vocab_file': '/mnt/work/training_data.tok4/1b/vocab/merged_vocab.txt',
					 'batch_size': 1024,
				   	 'num_steps': 1,
					 'cell_size': 200,
					 'num_layers': 2,
					 'max_vocab_size': -1,#20000,
					 'embedding_keep_prob': .9,
					 'rnn_dropout_keep_prob': .9,
					 'all_lowercase': True,
					 'mini_batches_between_checkpoint': 50,
					 'inference_output_node': 'output_logits_sm',
					 'learning_rate_decay': 0.95,
					 'learning_rate_decay_start_epoch': 5,
					 'max_grad_norm': 5,
					 'learning_rate': 1.0,
					 'optimizer':'sgd',
#					 'cell_activation': tf.sigmoid,
					 'use_single_sm': True,
					 'cell_type': 'BasicLSTM',  # GRU, BasicLSTM, BlockLSTM
}


