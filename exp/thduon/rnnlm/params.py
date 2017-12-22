import os


model_name = 'rnnlmV6'
params = { 'model_name': model_name,
					 'output_location': './output/%s/'%model_name,
					 'training_data_dir': '/mnt/work/tokenized_1b/',
					 'vocab_file': '../data/lowercase_1b_vocab.txt',
					 'batch_size': 256,
				   'num_steps': 35,
					 'cell_size': 512,
					 'num_layers': 2,
					 'max_vocab_size': -1,#20000,
					 'embedding_keep_prob': .75,
					 'rnn_dropout_keep_prob': .75,
					 'all_lowercase': True,
					 'mini_batches_between_checkpoint': 50,
					 'min_vocab_freq': 300,
					 'inference_output_node': 'output_logits_sm',
					 'learning_rate': .001,
					 'learning_rate_decay': 0.95,
					 'learning_rate_decay_start_epoch': 5,
					 'cell_type': 'BasicLSTM',  # GRU, BasicLSTM, BlockLSTM
}
