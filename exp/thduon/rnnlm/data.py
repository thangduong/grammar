import os

class RnnLmData:
	def __init__(self, params = {}, indexer=None):
		data_dir = params['training_data_dir']
		self._files = [os.path.join(data_dir, x) for x in os.listdir(data_dir)]
		self._file_index = -1
		self._current_file = None
		self._current_sentences = self.next_file()
		self._streams = []
		self._indexer = indexer
		self._unk_token = 'unk'
		self._current_epoch = 0
		self._current_index = 0
		self._num_steps = params['num_steps']

	def current_epoch(self):
		return self._current_epoch

	def current_index(self):
		return self._current_index

	def next_file(self):
		self._file_index += 1
		self._current_file = open(self._files[self._file_index], 'r', encoding='utf-8')
		for sentence in self._current_file:
			tokens = ['<s>'] + sentence.split() + ['</s>']
			if self._indexer is not None:
				_,tokens,_,_ = self._indexer.index_wordlist(tokens, unk_word=self._unk_token)
			yield tokens

	def next_sentence(self):
		return next(self._current_sentences)

	def next_batch(self, batch_size=5):
		num_steps = self._num_steps
		if len(self._streams) < batch_size:
			for x in range(len(self._streams),batch_size):
				self._streams.append([])
		batch_x = []
		batch_y = []
		batch_w = []
		for i in range(batch_size):
			steps_filled = 0
			x = [0] * num_steps
			y = [0] * num_steps

			# w is for importance sampling
			w = [0] * num_steps
			while steps_filled < num_steps:
				if self._streams[i] is None or len(self._streams[i]) <= 1:
					self._streams[i] = self.next_sentence()
				steps_to_fill = min(len(self._streams[i])-1,num_steps-steps_filled)
				x[steps_filled:steps_filled + steps_to_fill] = self._streams[i][:steps_to_fill]
				y[steps_filled:steps_filled + steps_to_fill] = self._streams[i][1:steps_to_fill+1]
				w[steps_filled:steps_filled + steps_to_fill] = [1.0]*steps_to_fill
				self._streams[i] = self._streams[i][steps_to_fill:]
				steps_filled += steps_to_fill
			batch_x.append(x)
			batch_y.append(y)
			batch_w.append(w)
		return {'x': batch_x, 'y': batch_y, 'w': batch_w }

if __name__ == "__main__":
	x = RnnLmData()
	b = x.next_batch()
	b = x.next_batch()
	print(b)
