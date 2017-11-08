from classifier_data import ClassifierData

params = { 'num_words_before': 5,
					 'num_words_after': 5,
					 'embedding_size': 300,
					 'vocab_size': 100000,
					 'embedding_device': None,
					 'batch_size': 128,
					 'num_classes': 2,
					 'mini_batches_between_checkpoint': 100,
					 'monolingual_dir': '/mnt/work/1-billion-word-language-modeling-benchmark'
					 }
test_data = ClassifierData.get_monolingual_test(params=params)
print(test_data.next_batch(batch_size=16))
