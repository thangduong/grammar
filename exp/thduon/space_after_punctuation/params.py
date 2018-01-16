import os

model_name = 'sapV0'
params = { 'model_name': model_name,
					 'output_location': './output/%s/'%model_name,
					 'num_chars_before': 15,
					 'num_chars_after': 15,
					 'embedding_size': 100,
					 'punctuations': [',','.',';',':'],
					 'enable_regularization': True,
					 'training_data_dir': '/mnt/work/training_data.tok4/',
					 }
