import os
import random
import framework.utils.common as utils
import math

def gen_data(tokens, keywords,
						 num_before=5, num_after=5,
						 pad_tok="<pad>", null_sample_factor=0):
	tokens = [pad_tok] * num_before + tokens + [pad_tok]*num_after
	n0 = []
	n1 = []
	if null_sample_factor < 0:
		null_sample_factor = 1/len(keywords)
	for toki, tok in enumerate(tokens):
		if tok.lower() in keywords:
			idx = keywords.index(tok.lower())
			before_idx = toki - num_before
			after_idx = toki + num_after + 1
			#yield [tokens[before_idx:toki] + tokens[toki+1:after_idx], idx + 1]
			n1.append([tokens[before_idx:toki] + tokens[toki+1:after_idx], idx + 1])#tokens[before_idx:toki] + tokens[toki+1:after_idx])
		elif toki > num_before and toki < len(tokens)-num_after:
			before_idx = toki - num_before
			after_idx = toki + num_after
			n0.append(tokens[before_idx:toki] + tokens[toki:after_idx])
			#yield [tokens[before_idx:toki] + tokens[toki+1:after_idx], 0]
	if len(n1) > 0:
		if null_sample_factor > 0:
			n0 = random.sample(n0, math.ceil(len(n1)*null_sample_factor))
#		n0 = random.sample(n0, math.ceil(len(n1)))
		n = []
		n += [[x,0] for x in n0]
		n += [[x,y] for x,y in n1]
		random.shuffle(n)
		for x in n:
			yield x

def gen_data_from_file(filename, keywords=[','], num_before=5, num_after=5, pad_tok="<pad>"):
	with open(filename) as f:
		for line in f:
			line = line.rstrip().lstrip()
			tokens = line.split()
			yield from gen_data(tokens, keywords=keywords, num_before=num_before, num_after=num_after, pad_tok=pad_tok)

class ClassifierData:
	def __init__(self, file_list, indexer=None, params=None):
		self._file_list = file_list
		self._cur_list = []
		self._next_file = 0
		self._keywords = utils.get_dict_value(params, 'keywords', [])
		self._num_before = utils.get_dict_value(params, 'num_words_before', 5)
		self._num_after = utils.get_dict_value(params, 'num_words_after', 5)
		self.load_next_file()
		self._indexer = indexer
		self._current_epoch = 0
		self._current_index = 0

	def load_next_file(self):
		print(self._file_list[self._next_file])
		self._cur_list = gen_data_from_file(self._file_list[self._next_file],
																				self._keywords,
																				self._num_before,
																				self._num_after)
		self._next_file += 1
		self._next_file %= len(self._file_list)

	def next_batch(self, batch_size=2, params=None):
		batch_x = []
		batch_y = []
		while (len(batch_y) < batch_size):
			try:
				rec = next(self._cur_list)
				if self._indexer is None:
					batch_x.append(rec[0])
				else:
					_, rec_indexed, _, _ = self._indexer.index_wordlist(rec[0])
					batch_x.append(rec_indexed)
				batch_y.append(rec[1])
				self._current_index += 1
			except StopIteration:
				self.load_next_file()
			except Exception as e:
				raise
				# no more records
		return {'sentence':batch_x, 'y': batch_y}

	def current_epoch(self):
		return self._current_epoch

	def current_index(self):
		return self._current_index

	@staticmethod
	def get_monolingual_training(base_dir = '/mnt/work/1-billion-word-language-modeling-benchmark', indexer=None, params=None):
		sub_path = 'training-monolingual.tokenized.shuffled'
		data_files = os.listdir(os.path.join(base_dir, sub_path))
		data_files = [os.path.join(base_dir, sub_path, x) for x in data_files]
		return ClassifierData(file_list = data_files, indexer=indexer, params=params)

	@staticmethod
	def get_monolingual_test(base_dir = '/mnt/work/1-billion-word-language-modeling-benchmark', indexer=None, params=None):
		sub_path = 'heldout-monolingual.tokenized.shuffled'
		data_files = os.listdir(os.path.join(base_dir, sub_path))
		data_files = [os.path.join(base_dir, sub_path, x) for x in data_files]
		return ClassifierData(file_list = data_files, indexer=indexer, params=params)


#result = gen_data_from_file('/mnt/work/1-billion-word-language-modeling-benchmark/training-monolingual.tokenized.shuffled/news.en-00001-of-00100')
#for x in result:
#	print(x)
if __name__ == "__main__":
	training_data = ClassifierData.get_monolingual_training()
	print(training_data.next_batch(batch_size=16))





