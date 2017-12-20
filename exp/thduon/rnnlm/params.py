import os


model_name = 'rnnlmV0'
params = { 'model_name': model_name,
					 'output_location': './output/%s/'%model_name,
					 'training_data_dir': '/mnt/work/tokenized_training_data/',
					 'vocab_file': '../data/filtered_lowercase_vocab.txt',
					 'batch_size': 32,
           'num_steps': 20,
					 'cell_size': 256,
					 'num_layers': 2,
					 'max_vocab_size': 20000,
					 'embedding_keep_prob': 0.9,
					 'rnn_dropout_keep_prob': 0.9,
}
