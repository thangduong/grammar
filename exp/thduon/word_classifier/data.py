import os
import random
import framework.utils.common as utils
import math

def merge_tokens_for_text(tokens):
	text = ''
	punctuation = [',', '.', ';']
	loc = []
	for toki, tok in enumerate(tokens):
		if len(text) > 0 and not tok in punctuation:
			text += ' '
		loc.append(len(text))
		text += tok
	return text, loc

#def gen_for_testing(tokens, keywords):
#	for toki, tok in enumerate(tokens):
#		if tok in keywords:


def gen_data(tokens, keywords,
						 num_before=5, num_after=5,
						 pad_tok="<pad>", null_sample_factor=0,
						 add_redundant_keyword_data=True,
						 use_negative_only_data=True,
						 add_keyword_removal_data=False):
	# assume incoming tokens constitute a correct sentences
	tokens = [pad_tok] * num_before + tokens + [pad_tok]*num_after
	n0 = [] # set of sentences that need [keyword]
	n1 = [] # set of sentences that need a keyword
	n2 = [] # set of sentences that need [keyword] because there is one there already
	n3 = [] # there is a [keyword] and it should be removed
	if null_sample_factor < 0:
		null_sample_factor = 1/len(keywords)
	#print(tokens)
	for toki, tok in enumerate(tokens):
		if tok.lower() in keywords:
			idx = keywords.index(tok.lower())
			before_idx = toki - num_before
			after_idx = toki + num_after + 1
			#yield [tokens[before_idx:toki] + tokens[toki+1:after_idx], idx + 1]
			n1.append([tokens[before_idx:toki] + tokens[toki+1:after_idx], idx + 1])#tokens[before_idx:toki] + tokens[toki+1:after_idx])
			n2.append(tokens[(before_idx+1):after_idx])
		elif toki > num_before and toki < len(tokens)-num_after:
			before_idx = toki - num_before
			after_idx = toki + num_after
			n0.append(tokens[before_idx:toki] + tokens[toki:after_idx])
			#yield [tokens[before_idx:toki] + tokens[toki+1:after_idx], 0]
			for keyword in keywords:
				n3.append(tokens[before_idx:toki] + [keyword] + tokens[toki:after_idx-1])
	if len(n1) > 0:
		if null_sample_factor > 0:
			n0 = random.sample(n0, math.ceil(len(n1)*null_sample_factor))
#		n0 = random.sample(n0, math.ceil(len(n1)))
		n = []
		n += [[x,0] for x in n0]
		n += [[x,y] for x,y in n1]
		if add_redundant_keyword_data:
			n += [[x,0] for x in n2]
		random.shuffle(n)
		for x in n:
			#print(x)
			yield x
	elif use_negative_only_data:
		n = []
		n += [[x,0] for x in n0]
		if add_redundant_keyword_data:
			n += [[x,0] for x in n2]
		random.shuffle(n)
		for x in n:
			#print(x)
			yield x


def gen_data_from_file(filename, keywords=[','], num_before=5, num_after=5, pad_tok="<pad>", null_sample_factor=0):
	with open(filename) as f:
		for line in f:
			line = line.rstrip().lstrip()
			tokens = line.split()
			yield from gen_data(tokens, keywords=keywords,
													num_before=num_before, num_after=num_after,
													pad_tok=pad_tok, null_sample_factor=null_sample_factor)

class ClassifierData:
	def __init__(self, file_list, indexer=None, params=None):
		self._file_list = file_list
		self._cur_list = []
		self._next_file = 0
		self._keywords = utils.get_dict_value(params, 'keywords', [])
		self._num_before = utils.get_dict_value(params, 'num_words_before', 5)
		self._num_after = utils.get_dict_value(params, 'num_words_after', 5)
		self._null_sample_factor = utils.get_dict_value(params, 'null_sample_factor', 0)
		self.load_next_file()
		self._indexer = indexer
		self._current_epoch = 0
		self._current_index = 0
		self._num_minibatches = 0

	def load_next_file(self):
		print(self._file_list[self._next_file])
		self._cur_list = gen_data_from_file(self._file_list[self._next_file],
																				keywords=self._keywords,
																				num_before=self._num_before,
																				num_after=self._num_after,
																				null_sample_factor=self._null_sample_factor)
		self._next_file += 1
		self._next_file %= len(self._file_list)
		if self._next_file == 0:
			self._current_epoch += 1

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
		self._num_minibatches += 1
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
	x = "We went to the store , and I bought some fruits".split()
	print(x)
	y = gen_data(x, [','])
	for yy in y:
		print(yy)
#	training_data = ClassifierData.get_monolingual_training()
#	print(training_data.next_batch(batch_size=16))
#	x = ['fraud', 'or', 'wrongdoing', 'have', 'contributed', 'to', 'the', 'current', 'problems', ';', 'authorities', 'need', 'to', ',', 'and', 'are', 'prosecuting', 'them', '.', '<pad>']
#	a,b = merge_tokens_for_text(x)
#	print(a)
#	print(b)





