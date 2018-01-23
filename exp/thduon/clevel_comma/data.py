import os
import random
import framework.utils.common as utils

max_value = 0

def _gen_data(sentence, num_before, num_after, null_sample_factor=0):
	global max_value
	slen = len(sentence)
	z = [ord(x) for x in sentence if ord(x)>255]
	if len(z)>0 and max(z) > max_value:
		max_value = max(z)
		print('max_value = %s'%max_value)
	sentence = [min(ord(x),255) for x in sentence]
	sentence = [0]*(num_before-1) + [1] + sentence + [2] + [0]*(num_after - 1)

	null_list = []
	pos_list = []
	keychars = [ord(',')]
	for i in range(num_before, num_before+slen):
		if sentence[i] in keychars:
			pos_list.append(sentence[i-num_before:i] + sentence[i+1:i+num_after+1])
		null_list.append(sentence[i-num_before:i] + sentence[i:i+num_after])

	if null_sample_factor<0:
		random.shuffle(null_list)
		null_list = null_list[:((len(pos_list)+1))]
	elif null_sample_factor>0:
		random.shuffle(null_list)
		null_list = null_list[:(null_sample_factor * (len(pos_list) + 1))]
	result = [[x,0] for x in null_list] + [[x,1] for x in pos_list]
	for x in result:
		yield x

def _gen_data_from_file(filename, num_before, num_after, null_sample_factor=0):
	with open(filename) as f:
		for line in f:
			line = line.rstrip().lstrip()
			yield from _gen_data(line, num_before, num_after, null_sample_factor=null_sample_factor)
			yield from _gen_data(line.lower(), num_before, num_after, null_sample_factor=null_sample_factor)

class CharLevelData:
	def __init__(self, file_list, params):
		self._file_list = file_list
		self._cur_list = []
		self._next_file = 0
		self._params = params
		self._keywords = utils.get_dict_value(params, 'keywords', [])
		self._num_before = utils.get_dict_value(params, 'num_before', 5)
		self._num_after = utils.get_dict_value(params, 'num_after', 5)
		self._null_sample_factor = utils.get_dict_value(params, 'null_sample_factor', 0)
		self._ignore_negative_data = utils.get_dict_value(params, 'ignore_negative_data', False)
		self._use_negative_only_data = utils.get_dict_value(params, 'use_negative_only_data', True)
		self._add_redundant_keyword_data = utils.get_dict_value(params, 'add_redundant_keyword_data', True)
		self._start_token = utils.get_dict_value(params, 'start_token', None)
		self._null_sample_factor = utils.get_dict_value(params, 'null_sample_factor')
		self.load_next_file()
		self._current_epoch = 0
		self._current_index = 0
		self._num_minibatches = 0
		self._dump_num_batches = 10
		self._mimibatch_dump_dir = utils.get_dict_value(params, 'output_location', '.')
		self._y_count = [0]*utils.get_dict_value(params,'num_classes',2)
		self._count_y = True


	def load_next_file(self):
		print(self._file_list[self._next_file])
		self._cur_list = _gen_data_from_file(self._file_list[self._next_file], self._num_before,
																				 self._num_after, null_sample_factor=self._null_sample_factor)
		self._next_file += 1
		self._next_file %= len(self._file_list)
		if self._next_file == 0:
			self._current_epoch += 1

	def next_batch(self, batch_size=2, params=None):
		batch_x = []
		batch_y = []

		if self._dump_num_batches > 0:
			mb_dump_file = open(os.path.join(self._mimibatch_dump_dir, "mb_dump_%05d.txt"%self._dump_num_batches ),'w')
			self._dump_num_batches-=1
		else:
			mb_dump_file = None

		while (len(batch_y) < batch_size):
			try:
				rec = next(self._cur_list)
				if mb_dump_file is not None:
					mb_dump_file.write('%03d %s\n'%(rec[1], ''.join([chr(x) for x in rec[0]])))
				batch_x.append(rec[0])
				batch_y.append(rec[1])
				self._current_index += 1
			except StopIteration:
				self.load_next_file()
			except Exception as e:
				raise
				# no more records
		self._num_minibatches += 1
		result = {'sentence':batch_x, 'y': batch_y}


		if mb_dump_file is not None:
			mb_dump_file.close()

		if self._count_y:
			for classi in range(len(self._y_count)):
				self._y_count[classi] += batch_y.count(classi)
		return result

	def current_epoch(self):
		return self._current_epoch

	def current_index(self):
		return self._current_index

	@staticmethod
	def get_monolingual_training(base_dir = '/mnt/work/1-billion-word-language-modeling-benchmark', params=None):
		sub_path = 'training-monolingual'
		data_files = os.listdir(os.path.join(base_dir, sub_path))
		data_files = [os.path.join(base_dir, sub_path, x) for x in data_files]
		return CharLevelData(file_list = data_files, params=params)
	@staticmethod
	def get_training_data(base_dir = '/mnt/work/1-billion-word-language-modeling-benchmark', params=None):
		data_files = os.listdir(base_dir)
		data_files = [os.path.join(base_dir, x) for x in data_files]
		return CharLevelData(file_list = data_files, params=params)

if __name__ == "__main__":
#	x = "We went to the store , and I bought some fruits"
#	r = list(_gen_data(x,10,10))

#	for u, v in r:
#		y = [chr(a) for a in u]
#		print("%s %s"%(v, y))

	params = utils.load_param_file('params.py')
	d = CharLevelData.get_monolingual_training(params=params)
	print(d.next_batch(batch_size=16))
#	x = ['fraud', 'or', 'wrongdoing', 'have', 'contributed', 'to', 'the', 'current', 'problems', ';', 'authorities', 'need', 'to', ',', 'and', 'are', 'prosecuting', 'them', '.', '<pad>']
#	a,b = merge_tokens_for_text(x)
#	print(a)
#	print(b)





