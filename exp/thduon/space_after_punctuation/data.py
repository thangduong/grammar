from framework.base_data import BaseDataObject
import framework.utils.common as utils
import os
import random

def generate_data_from_sentence(params, sentence):
	punctuations = params['punctuations']
	num_chars_before = params['num_chars_before']
	num_chars_after = params['num_chars_after']
	vocab_size = params['vocab_size']
	space_before = utils.get_dict_value(params, 'space_before', False) # if not before, then after
	ord_sentence = [0]*num_chars_before + [min(ord(x), vocab_size-1) for x in sentence] + [0]*num_chars_after

	for i in range(num_chars_before, num_chars_before + len(sentence) - 2):
		if chr(ord_sentence[i]) in punctuations:
			if space_before:
				after = ord_sentence[(i):(i+num_chars_after)]
				if ord_sentence[i-1] == ord(' '):
					before = ord_sentence[(i - num_chars_before - 1):(i-1)]
					# yes, space
					yield before + after, 1
				# no, no space
				before = ord_sentence[(i - num_chars_before):(i)]
				yield before + after, 0
			else:
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
	case_mode = utils.get_dict_value(params, 'case_mode', 'normal')
	preserve_line_start = utils.get_dict_value(params, 'preserve_line_start', False)
	with open(filepath, 'r') as f:
		lines = []
		for i, line in enumerate(f):
			line = line.rstrip().lstrip()
			lines.append(line)
			if ((i+1)%lines_per_group)==0:
#				print(len(lines))
				if not preserve_line_start:
					random.shuffle(lines)
					sentences = [' '.join(lines)]
				else:
					sentences = []
					for j, cline in enumerate(lines):
						clines = lines[:j]+lines[(j+1):]
						random.shuffle(clines)
						sentences += [' '.join([cline]+clines)]
#					print(len(sentences))
#					for s in sentences:
#						print('-%s'%s)
#					exit(0)
				lines = []
				for sentence in sentences:
					if case_mode == 'normal':
						yield from generate_data_from_sentence(params, sentence)
					elif case_mode == 'normal_and_lower':
						yield from generate_data_from_sentence(params, sentence)
						yield from generate_data_from_sentence(params, sentence.lower())
					elif case_mode == 'lower':
						yield from generate_data_from_sentence(params, sentence.lower())


class Data(BaseDataObject):
	def __init__(self, params, filelist):
		super().__init__()
		self._params = params
		self._current_file = -1
		self._current_epoch = -1
		self._filelist = filelist
		self._debug_write_count = 0
		self._debug_write_count_max = 1000000
		self._output_dump_dir = utils.get_dict_value(params, 'output_location', '.')
		self._debug_file = open(os.path.join(self._output_dump_dir, 'minibatch_dump.txt'),'w')
		self._processed_files = open(os.path.join(self._output_dump_dir, 'processed_files.txt'),'w')
		random.shuffle(self._filelist)
		self.advance_file()

	def advance_file(self):
		self._current_file += 1
		if self._current_file == 0:
			self._current_epoch += 1
		self._processed_files.write("%s %s %s\r\n"%(self._current_epoch,
																								self._current_file,
																								self._filelist[self._current_file]))
		self._processed_files.flush()
		self._data = generate_data_from_file(self._params, self._filelist[self._current_file])
		if (self._current_file == len(self._filelist)):
			self._current_file = -1
			random.shuffle(self._filelist)

	def next_batch(self, batch_size, params=None):
		records_read = 0

		sentences = []
		need_space = []
		while (records_read < batch_size):
			try:
				x,y = next(self._data)
				if self._debug_file is not None and self._debug_write_count < self._debug_write_count_max:
					self._debug_file.write("%s: %s\r\n"%(y,''.join([chr(a) for a in x])))
					self._debug_file.flush()
					self._debug_write_count += 1
				elif self._debug_file is not None:
					self._debug_file.close()
					self._debug_file = None
				sentences.append(x)
				need_space.append(y)
				records_read += 1
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
	sentence = "Hello (1,2,3)"
	params = utils.load_param_file('params.py')
	data = generate_data_from_sentence(params,sentence)
	#data = generate_data_from_file(params, '/mnt/work/training-monolingual/news.2007.en.shuffled')
	for x,y in data:
		print('%s %s->%s'%(len(x), [chr(a) for a in x],y))
#	d = Data.get_data(params, '/mnt/work/training-monolingual' )
#	print(d.next_batch(5))
