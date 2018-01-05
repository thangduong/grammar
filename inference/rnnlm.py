# inference side for RNN LM model

class RnnLm:
	def __init__(self):
		None

	def load(self, model_dir):
		self._model_dir = model_dir
		self._paramsfile = os.path.join(self._model_dir, 'params.py')
		self._params = utils.load_param_file(self._paramsfile)
		self._num_before = utils.get_dict_value(self._params, "num_words_before")
		self._num_after = utils.get_dict_value(self._params, "num_words_after")
		ckpt = os.path.join(utils.get_dict_value(self._params,'output_location'),
												utils.get_dict_value(self._params, 'model_name') + '.ckpt')
		self._e = Evaluator.load2(ckpt)

		vocab_file = os.path.join(utils.get_dict_value(self._params, 'output_location'), 'vocab.pkl')
		if os.path.exists(vocab_file):
			self._i = TextIndexer.from_file(vocab_file)
#		self._keywords = self._params['keywords']
#		self._id_to_word = self._params['id_to_keyword']

	def get_model_name(self):
		return self._params['model_name']


if __name__ == "__main__":
	x = RnnLmData()
	b = x.next_batch()
	b = x.next_batch()
	print(b)
