import os
params = { 'model_name': 'commav0',
					 'output_location': './output/',
					 'null_sample_factor': 0,  # <0= equal null as non null per sentence, 0 = don't do anything, >0 = factor
					 'num_words_before': 10,
					 'num_words_after': 10,
					 'embedding_size': 300,
					 'embedding_device': None,
					 'batch_size': 8192,
					 'keywords': [','],
					 'mini_batches_between_checkpoint': 100,
					 'monolingual_dir': '/mnt/work/1-billion-word-language-modeling-benchmark',
					 'vocab_file': '/mnt/work/1-billion-word-language-modeling-benchmark/1b_word_vocab.txt'
					 }
