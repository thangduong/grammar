import os
import random
import framework.utils.common as utils
import math

def merge_tokens_for_text(tokens):
	text = ''
	punctuation = [',', '.', ';', "'s", "n't", "/", ')', '(', '[', ']']
	no_space_after_toks = ["/"]
	loc = []
	no_space_after = False
	quotes = ['"', "'"]
	quote_count = [0,0]
	in_quote = False
	for toki, tok in enumerate(tokens):
		quote_tok = False
		if text in quotes:
			quote_tok = True
			quote_count[quotes.index(text)] += 1
			if (quote_count[quotes.index(text)]%2) == 1:
				in_quote = True
			else:
				in_quote = False
		if len(text) > 0 \
				and (not tok in punctuation) \
				and (not no_space_after) \
				:
			text += ' '
		if tok in no_space_after_toks:
			no_space_after = True
		else:
			no_space_after = False
		loc.append(len(text))
		text += tok
	return text, loc

def _gen_data(dataobj, tokens, keywords,
						 num_before=5, num_after=5,
						 pad_tok="<pad>", null_sample_factor=0,
						 add_redundant_keyword_data=True,
						 use_negative_only_data=True,
						 ignore_negative_data=False,
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
	class_offset = 1
	if ignore_negative_data:
		class_offset = 0
	for toki, tok in enumerate(tokens):
		if tok.lower() in keywords:
			idx = keywords.index(tok.lower())
			before_idx = toki - num_before
			after_idx = toki + num_after + 1
			#yield [tokens[before_idx:toki] + tokens[toki+1:after_idx], idx + 1]
			n1.append([tokens[before_idx:toki] + tokens[toki+1:after_idx], idx + class_offset])#tokens[before_idx:toki] + tokens[toki+1:after_idx])
			n2.append(tokens[(before_idx + 1):after_idx])
			n2.append(tokens[(before_idx):(after_idx-1)])
		elif toki > num_before and toki < len(tokens)-num_after and not ignore_negative_data:
			before_idx = toki - num_before
			after_idx = toki + num_after
			n0.append(tokens[before_idx:toki] + tokens[toki:after_idx])
			#yield [tokens[before_idx:toki] + tokens[toki+1:after_idx], 0]
			for keyword in keywords:
				n3.append(tokens[before_idx:toki] + [keyword] + tokens[toki:after_idx-1])
	if len(n1) > 0:
		if ignore_negative_data:
			n0 = []
		else:
			if null_sample_factor > 0:
				n0 = random.sample(n0, min(len(n0),math.ceil(len(n1)*null_sample_factor)))
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
	elif use_negative_only_data and not ignore_negative_data:
		n = []
		n += [[x,0] for x in n0]
		if add_redundant_keyword_data:
			n += [[x,0] for x in n2]
		random.shuffle(n)
		for x in n:
			#print(x)
			yield x


def _gen_data_from_file(dataobj, filename, keywords=[','], num_before=5, num_after=5,
											 pad_tok="<pad>", null_sample_factor=0,
											 use_negative_only_data=True,
											 add_redundant_keyword_data=True,
											 start_token=None,
											 ignore_negative_data=False,
											 all_lowercase=False,
											 gen_data_fcn = _gen_data):
	with open(filename) as f:
		for line in f:
			line = line.rstrip().lstrip()
			if all_lowercase:
				line = line.lower()
			tokens = line.split()
			if start_token is not None and len(start_token)>0:
				tokens = [start_token] + tokens
			yield from gen_data_fcn(dataobj, tokens, keywords=keywords,
													num_before=num_before, num_after=num_after,
													pad_tok=pad_tok, null_sample_factor=null_sample_factor,
													add_redundant_keyword_data=add_redundant_keyword_data,
													use_negative_only_data=use_negative_only_data,
													ignore_negative_data=ignore_negative_data)

class ClassifierData:
	def __init__(self, file_list, indexer=None, params=None,
							 gen_data_from_file_fcn = _gen_data_from_file,
							 gen_data_fcn = _gen_data):
		self._gen_data_from_file_fcn = gen_data_from_file_fcn
		self._gen_data_fcn = gen_data_fcn
		self._file_list = file_list
		self._cur_list = []
		self._next_file = 0
		self._params = params
		_keywords = utils.get_dict_value(params, 'keywords', [])
		if utils.get_dict_value(params, 'keywords_as_map', False):
			self._keywords = {}
			for i, k in enumerate(_keywords):
				self._keywords[k] = i
		else:
			self._keywords = _keywords

		self._num_before = utils.get_dict_value(params, 'num_words_before', 5)
		self._num_after = utils.get_dict_value(params, 'num_words_after', 5)
		self._null_sample_factor = utils.get_dict_value(params, 'null_sample_factor', 0)
		self._ignore_negative_data = utils.get_dict_value(params, 'ignore_negative_data', False)
		self._use_negative_only_data = utils.get_dict_value(params, 'use_negative_only_data', True)
		self._add_redundant_keyword_data = utils.get_dict_value(params, 'add_redundant_keyword_data', True)
		self._start_token = utils.get_dict_value(params, 'start_token', None)
		self._use_char_cnn = utils.get_dict_value(params, 'use_char_cnn', False)
		self._all_lowercase = utils.get_dict_value(params, 'all_lowercase', False)
		self._ccnn_num_words = utils.get_dict_value(params, 'ccnn_num_words', 0)  # 0 = 1st or 2nd word after, otherwise the # of words after delimited by 0
		self._ccnn_word_len = utils.get_dict_value(params, 'word_len')
		self._mimibatch_dump_dir = utils.get_dict_value(params, 'output_location', '.')
		self._y_count = [0]*utils.get_dict_value(params,'num_classes',2)
		if self._all_lowercase:
			if self._start_token is not None:
				self._start_token = self._start_token.lower()
		self._num_files_processed = -1
		self.load_next_file()
		self._indexer = indexer
		self._current_epoch = 0
		self._current_index = 0
		self._num_minibatches = 0
		self._dump_num_batches = 10
		self._verbose = False
		self._count_y = True

	def load_next_file(self):
		print(self._file_list[self._next_file])
		self._num_files_processed += 1
		self._cur_list = self._gen_data_from_file_fcn(self, self._file_list[self._next_file],
																				keywords=self._keywords,
																				num_before=self._num_before,
																				num_after=self._num_after,
																				null_sample_factor=self._null_sample_factor,
																				ignore_negative_data=self._ignore_negative_data,
																				add_redundant_keyword_data=self._add_redundant_keyword_data,
																				use_negative_only_data=self._use_negative_only_data,
																				start_token=self._start_token,
																			  all_lowercase=self._all_lowercase,
																				gen_data_fcn=self._gen_data_fcn)
		self._next_file += 1
		self._next_file %= len(self._file_list)
		if self._num_files_processed == len(self._file_list):
			self._num_files_processed = 0
			self._current_epoch += 1

	def next_batch(self, batch_size=2, params=None):
		batch_x = []
		batch_y = []
		if self._use_char_cnn:
			batch_ccnn = []


		if self._dump_num_batches > 0:
			mb_dump_file = open(os.path.join(self._mimibatch_dump_dir, "mb_dump_%05d.txt"%self._dump_num_batches ),'w')
			self._dump_num_batches-=1
		else:
			mb_dump_file = None

		total_unk = 0
		all_unk = []
		total_indexed = 0
		while (len(batch_y) < batch_size):
			try:
				rec = next(self._cur_list)
				if mb_dump_file is not None:
					mb_dump_file.write('%03d %s\n'%(rec[1], rec[0]))
				tok0 = rec[0][int(len(rec[0]) / 2)]
				# 0 = 1st or 2nd word after, otherwise the # of words after delimited by 0
				if self._ccnn_num_words == 0:
					if len(tok0) == 1:
						tok0 = rec[0][int(len(rec[0]) / 2) + 1]
				else:
					for i in range(self._ccnn_num_words-1):
						#tok0 += [0] + rec[0][int(len(rec[0]) / 2) + 1 + i]
						tok0 += chr(0) + rec[0][int(len(rec[0]) / 2) + 1 + i]
				if self._indexer is None:
					batch_x.append(rec[0])
#						batch_ccnn.append(tok0)
				else:
					num_indexed, rec_indexed, unk_count, unk_list = self._indexer.index_wordlist(rec[0])
					total_indexed += num_indexed
					total_unk += unk_count
					all_unk += unk_list
					batch_x.append(rec_indexed)

				# add the word input if use it
				if self._use_char_cnn:
					cinput = []
					for ci, ch in enumerate(tok0):
						if ci >= self._ccnn_word_len:
							break
						if ord(ch) >= 128:
							cinput.append(127)
						else:
							cinput.append(ord(ch))
					cinput += [0] * (self._ccnn_word_len - len(cinput))
					batch_ccnn.append(cinput)
				batch_y.append(rec[1])
				if self._count_y:
					self._y_count[rec[1]] += 1
				self._current_index += 1
			except StopIteration:
				self.load_next_file()
			except Exception as e:
				raise
				# no more records
		self._num_minibatches += 1
		if self._use_char_cnn:
			result = {'sentence':batch_x, 'y': batch_y, 'word': batch_ccnn}
		else:
			result = {'sentence':batch_x, 'y': batch_y}

		if self._verbose:
			print("NUMBER OF UNK: %d (%0.2f)"%(total_unk, 100*total_unk / max(total_indexed,1)))
		if mb_dump_file is not None:
			mb_dump_file.close()
#		if self._dump_num_batches > 0:
#			with open('dump_batch_%02d.txt'%self._dump_num_batches,'w') as f:
#				for (x,y) in zip(result['sentence'], result['y']):
#					f.write('%02d: %s\n'%(y,x))
#			self._dump_num_batches -= 1
#		if self._count_y:
#			for classi in range(len(self._y_count)):
#				self._y_count[classi] += batch_y.count(classi)
		return result

	def current_epoch(self):
		return self._current_epoch

	def current_index(self):
		return self._current_index

	@staticmethod
	def get_training_data(base_dir='/mnt/work/toenized_training_data', indexer=None,
															 params=None,
															 gen_data_from_file_fcn=_gen_data_from_file,
															 gen_data_fcn=_gen_data):
		#		sub_path = 'alltrain'
		data_files = os.listdir(base_dir)
		data_files = [os.path.join(base_dir, x) for x in data_files]
		return ClassifierData(file_list=data_files, indexer=indexer, params=params,
													gen_data_from_file_fcn=gen_data_from_file_fcn,
													gen_data_fcn=gen_data_fcn)

	@staticmethod
	def get_monolingual_training(base_dir = '/mnt/work/1-billion-word-language-modeling-benchmark', indexer=None, params=None,
							 gen_data_from_file_fcn = _gen_data_from_file,
							 gen_data_fcn = _gen_data):
		sub_path = 'training-monolingual.tokenized.shuffled'
#		sub_path = 'alltrain'
		data_files = os.listdir(os.path.join(base_dir, sub_path))
		data_files = [os.path.join(base_dir, sub_path, x) for x in data_files]
		return ClassifierData(file_list = data_files, indexer=indexer, params=params,
							 gen_data_from_file_fcn = gen_data_from_file_fcn,
							 gen_data_fcn = gen_data_fcn)

	@staticmethod
	def get_monolingual_test(base_dir = '/mnt/work/1-billion-word-language-modeling-benchmark', indexer=None, params=None,
							 gen_data_from_file_fcn = _gen_data_from_file,
							 gen_data_fcn = _gen_data):
		sub_path = 'heldout-monolingual.tokenized.shuffled'
		data_files = os.listdir(os.path.join(base_dir, sub_path))
		data_files = [os.path.join(base_dir, sub_path, x) for x in data_files]
		return ClassifierData(file_list = data_files, indexer=indexer, params=params,
							 gen_data_from_file_fcn = gen_data_from_file_fcn,
							 gen_data_fcn = gen_data_fcn)

	@staticmethod
	def get_wiki_test(base_dir = '/mnt/work/1-billion-word-language-modeling-benchmark', indexer=None, params=None,
							 gen_data_from_file_fcn = _gen_data_from_file,
							 gen_data_fcn = _gen_data):
		sub_path = 'wiki'
		data_files = os.listdir(os.path.join(base_dir, sub_path))
		data_files = [os.path.join(base_dir, sub_path, x) for x in data_files]
		print(data_files)
		return ClassifierData(file_list = data_files, indexer=indexer, params=params,
							 gen_data_from_file_fcn = gen_data_from_file_fcn,
							 gen_data_fcn = gen_data_fcn)


#result = gen_data_from_file('/mnt/work/1-billion-word-language-modeling-benchmark/training-monolingual.tokenized.shuffled/news.en-00001-of-00100')
#for x in result:
#	print(x)
if __name__ == "__main__":
	from framework.utils.data.text_indexer import TextIndexer
#	x = "We went to the store , and I bought some fruits".split()
#	print(x)
#	y = gen_data(x, [','])
#	for yy in y:
#		print(yy)
	params = utils.load_param_file('../determiner.ccnn/params.py')
	indexer = TextIndexer.from_txt_file(utils.get_dict_value(params, 'vocab_file'),
																			max_size=utils.get_dict_value(params, 'max_vocab_size', -1))
	indexer.add_token('<pad>')
	indexer.add_token('unk')
	d = ClassifierData.get_monolingual_training(indexer=indexer, params=params)
	a = d.next_batch(batch_size=2)
	print(a)
#	print(training_data.next_batch(batch_size=16))
#	x = ['fraud', 'or', 'wrongdoing', 'have', 'contributed', 'to', 'the', 'current', 'problems', ';', 'authorities', 'need', 'to', ',', 'and', 'are', 'prosecuting', 'them', '.', '<pad>']
#	a,b = merge_tokens_for_text(x)
#	print(a)
#	print(b)





