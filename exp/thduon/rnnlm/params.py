import os


model_name = 'rnnlmV13'
params = { 'model_name': model_name,
					 'output_location': './output/%s/'%model_name,
					 'training_data_dir': '/mnt/work/training_data.tok2/1b',
					 'vocab_file': '/mnt/work/training_data.tok2/1b/vocab/merged_vocab.txt',
					 'batch_size': 128,
				   'num_steps': 20,
					 'cell_size': 1024,
					 'num_layers': 2,
					 'max_vocab_size': -1,#20000,
					 'embedding_keep_prob': .6,
					 'rnn_dropout_keep_prob': .6,
					 'all_lowercase': True,
					 'mini_batches_between_checkpoint': 50,
					 'inference_output_node': 'output_logits_sm',
					 'learning_rate': .001,
					 'learning_rate_decay': 0.95,
					 'learning_rate_decay_start_epoch': 5,
					 'max_grad_norm': 5,
					 'use_single_sm': True,
					 'cell_type': 'BasicLSTM',  # GRU, BasicLSTM, BlockLSTM
}


