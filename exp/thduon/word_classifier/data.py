import os
import random
import framework.utils.common as utils
import math

def has_alphanumeric(input_string):
	"""
	returns true if input_string has an alpha_numeric char
	:param input_string:
	:return: true if iput_string has an alpha_numeric char
	"""
	for x in input_string:
		if x.isalnum():
			return True
	return False

def fix_token(token):
	"""
	Some tokens are of the form <int>|1945 . For these tokens,
	it should split and return <int> and 1945.  Otherwise, it should
	return the token twice.
	@param token: incoming token
	@return: two pieces
	"""
	left = right = token
	if token[0] == "<":
		end = token.find('>')
		if end > 0 and end < len(token)-1 and token[end+1]=='|':
			left = token[:end+1]
			right = token[end+2:]
	return left, right

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
	last_tok = ""
	for toki, tok in enumerate(tokens):
		if dataobj._all_lowercase:
			tok_to_check = tok.lower()
		else:
			tok_to_check = tok
		if tok_to_check in keywords:


			idx = keywords.index(tok_to_check)
			before_idx = toki - num_before
			after_idx = toki + num_after + 1
			word_after = tokens[toki+1]
			n1.append([tokens[before_idx:toki] + [word_after] + tokens[toki+2:after_idx], idx + class_offset])#tokens[before_idx:toki] + tokens[toki+1:after_idx])

			if last_tok == dataobj._start_token \
				and dataobj._start_sentence_synthetic_capital_and_lowercase \
				and	not dataobj._all_lowercase:
				word_after = tokens[toki+1].title()
				n1.append([tokens[before_idx:toki] + [word_after] + tokens[toki+2:after_idx], idx + class_offset])#tokens[before_idx:toki] + tokens[toki+1:after_idx])

			n2.append(tokens[(before_idx + 1):after_idx])
			n2.append(tokens[(before_idx):(after_idx-1)])
		elif toki > num_before and toki < len(tokens)-num_after and not ignore_negative_data:
			before_idx = toki - num_before
			after_idx = toki + num_after
			n0.append(tokens[before_idx:toki] + tokens[toki:after_idx])
			#yield [tokens[before_idx:toki] + tokens[toki+1:after_idx], 0]
			for keyword in keywords:
				n3.append(tokens[before_idx:toki] + [keyword] + tokens[toki:after_idx-1])
		last_tok = tok

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
	with open(filename, 'r', encoding='utf-8', errors="ignore") as f:
		for line in f:
			line = line.rstrip().lstrip()
#			if all_lowercase:
#				line = line.lower()
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
		self._start_sentence_synthetic_capital_and_lowercase = utils.get_dict_value(params, 'start_sentence_synthetic_capital_and_lowercase', False)
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
		self._dump_dir = utils.get_dict_value(params, 'output_location', '.')
		if self._ignore_negative_data:
			num_classes = len(_keywords)
		else:
			num_classes = len(_keywords) + 1
		self._y_count = [0]*utils.get_dict_value(params,'num_classes',num_classes)
		self._lowercase_char_path = utils.get_dict_value(params, 'lowercase_char_path', True)
		self._ccnn_skip_nonalphanumeric = utils.get_dict_value(params, 'ccnn_skip_nonalphanumeric', False)

		if self._all_lowercase:
			if self._start_token is not None:
				self._start_token = self._start_token.lower()
		self._num_files_processed = -1
		self._processed_data_files_fp = open(os.path.join(self._dump_dir, "processed_data_files.txt"), 'w')
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
		self._processed_data_files_fp.write("%s\r\n"%self._file_list[self._next_file])
		self._processed_data_files_fp.flush();
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
			mb_dump_file = open(os.path.join(self._dump_dir, "mb_dump_%05d.txt" % self._dump_num_batches), 'w')
			self._dump_num_batches-=1
		else:
			mb_dump_file = None

		total_unk = 0
		all_unk = []
		total_indexed = 0
		while (len(batch_y) < batch_size):
			try:
				rec = next(self._cur_list)

				rec0_left, rec0_right = map(list, zip(*[fix_token(x) for x in rec[0]]))
				tok0 = '' #rec0_right[int(len(rec[0]) / 2)]
				# 0 = 1st or 2nd word after, otherwise the # of words after delimited by 0
				if self._ccnn_num_words == 0:
					if len(tok0) == 1:
						tok0 = rec0_right[int(len(rec[0]) / 2) + 1]
				else:
					words_copied = 0
					words_examined = 0
					while (words_copied < self._ccnn_num_words) and (words_examined + int(len(rec[0]) / 2) < len(rec0_right)):
						word = rec0_right[int(len(rec[0]) / 2) + words_examined]
						if word=='<pad>':
							break
						if not self._ccnn_skip_nonalphanumeric or has_alphanumeric(word): # _ccnn_skip_nonalphanumeric is not used right now
							if len(tok0)>0:
								tok0 += ' '
							tok0 += word
							words_copied += 1
						words_examined += 1
#					for i in range(self._ccnn_num_words-1):
#						tok0 += ' ' + rec0_right[int(len(rec[0]) / 2) + 1 + i]
				if self._use_char_cnn:
					if mb_dump_file is not None:
						mb_dump_file.write('[%s]\n' % tok0)
				if self._all_lowercase:
					test_sentence = [x.lower() for x in rec0_left]
				else:
					test_sentence = rec0_left
				if mb_dump_file is not None:
					mb_dump_file.write('%03d %s\n'%(rec[1], test_sentence))
				if self._lowercase_char_path:
					tok0 = tok0.lower()
				if self._indexer is None:
					batch_x.append(test_sentence)
#						batch_ccnn.append(tok0)
				else:
					num_indexed, rec_indexed, unk_count, unk_list = self._indexer.index_wordlist(test_sentence)
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
#					print(self._y_count)
#					print(rec[1])
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
	def get_data_from_dirs(dir_list, indexer=None, params=None, gen_data_from_file_fcn=_gen_data_from_file, gen_data_fcn=_gen_data):
		data_files = []
		for dir in dir_list:
			cur_data_files = os.listdir(dir)
			data_files += [os.path.join(dir, x) for x in cur_data_files if os.path.isfile(os.path.join(dir,x))]
		return ClassifierData(file_list=data_files, indexer=indexer, params=params,
													gen_data_from_file_fcn=gen_data_from_file_fcn,
													gen_data_fcn=gen_data_fcn)
	@staticmethod
	def get_data(params, type='training', indexer=None, params=None, gen_data_from_file_fcn=_gen_data_from_file, gen_data_fcn=_gen_data):
		data_dir = params['data_dir']
		if type == 'training':
			data_dir = os.path.join(data_dir, 'train')
		elif type == 'test':
			data_dir = os.path.join(data_dir, 'test')
		elif type == 'devtest':
			data_dir = os.path.join(data_dir, 'devtest')
		elif type == 'valid':
			data_dir = os.path.join(data_dir, 'valid')
		return ClassifierData.get_data_from_dirs([data_dir],
																						 indexer=indexer,
																						 params=params,
																						 gen_data_from_file_fcn=gen_data_from_file_fcn,
																						 gen_data_fcn=gen_data_fcn)

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
	indexer = None
	d = ClassifierData.get_data('/mnt/work/data_gen_test', indexer=indexer, params=params)
	a = d.next_batch(batch_size=20)
	for x,y in zip(a['y'],a['sentence']):
		print("%s:%s"%(y,x))
#	print(training_data.next_batch(batch_size=16))
#	x = ['fraud', 'or', 'wrongdoing', 'have', 'contributed', 'to', 'the', 'current', 'problems', ';', 'authorities', 'need', 'to', ',', 'and', 'are', 'prosecuting', 'them', '.', '<pad>']
#	a,b = merge_tokens_for_text(x)
#	print(a)
#	print(b)





