from framework.base_data import BaseDataObject
import framework.utils.common as utils
import os
import random

def generate_data_from_sentence(params, sentence):
	punctuations = params['punctuations']
	num_chars_before = params['num_chars_before']
	num_chars_after = params['num_chars_after']
	vocab_size = params['vocab_size']
	ord_sentence = [0]*num_chars_before + [min(ord(x), vocab_size-1) for x in sentence] + [0]*num_chars_after

	for i in range(num_chars_before, num_chars_before + len(sentence) - 2):
		if chr(ord_sentence[i]) in punctuations:
			before = ord_sentence[(i - num_chars_before + 1):(i+1)]
			if ord_sentence[i+1] == ord(' '):
				# yes, space
				after = ord_sentence[(i+2):(i+num_chars_after+2)]
				yield before + after, 1
			# no, no space
			after = ord_sentence[(i+1):(i+num_chars_after+1)]
			yield before + after, 0

def generate_data_from_file(params, filepath):
	lines_per_group = params['lines_per_group']
	with open(filepath, 'r') as f:
		lines = []
		for i, line in enumerate(f):
			line = line.rstrip().lstrip()
			lines.append(line)
			if (i%lines_per_group)==0:
				random.shuffle(lines)
				sentence = ' '.join(lines)
				lines = []
				yield from generate_data_from_sentence(params, sentence)


class Data(BaseDataObject):
	def __init__(self, params, filelist):
		super().__init__()
		self._params = params
		self._current_file = 0
		self._filelist = filelist
		self._debug_write_count = 0
		self._debug_file = open('test.txt','w')
		random.shuffle(self._filelist)
		self.advance_file()

	def advance_file(self):
		self._data = generate_data_from_file(self._params, self._filelist[self._current_file])
		self._current_file += 1
		if (self._current_file == len(self._filelist)):
			self._current_file = 0
			self._current_epoch += 1
			random.shuffle(self._filelist)

	def next_batch(self, batch_size, params=None):
		records_read = 0

		sentences = []
		need_space = []
		while (records_read < batch_size):
			try:
				x,y = next(self._data)
				sentences.append(x)
				need_space.append(y)
				records_read += 1
				if not (self._debug_file is None):
					self._debug_file.write('%s - %s\n'%(y,''.join([chr(a) for a in x])))
					self._debug_write_count += 1
					if self._debug_write_count > 1000:
						self._debug_file.close()
						self._debug_file = None
					else:
						self._debug_file.flush()
			except StopIteration:
				self.advance_file()

		self._current_index += records_read
		self._num_minibatches += 1
		return {'sentence': sentences, 'y': need_space}

	@staticmethod
	def get_data(params, base_dir='/mnt/work/training_data'):
		#		sub_path = 'alltrain'
		data_files = os.listdir(base_dir)
		data_files = [os.path.join(base_dir, x) for x in data_files if os.path.isfile(os.path.join(base_dir,x))]
		return Data(params, data_files)


if __name__ == '__main__':
	sentence = "Hello, world! This is a test. There are 10,000 people on earth."
	params = utils.load_param_file('params.py')
	#data = generate_data(sentence, params)
	#data = generate_data_from_file(params, '/mnt/work/training-monolingual/news.2007.en.shuffled')
	#for x,y in data:
	#	print('%s %s->%s'%(len(x), [chr(a) for a in x],y))
	d = Data.get_data(params, '/mnt/work/training-monolingual' )
	print(d.next_batch(5))