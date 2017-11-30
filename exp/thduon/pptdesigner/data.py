import framework.utils.common as utils
import numpy as np
import os


class PPTDesignerData:
	def __init__(self, params=None,
							 files=[['features_000.npy', 'scores_000.npy'],
											['features_001.npy', 'scores_001.npy'],
											['features_002.npy', 'scores_002.npy'],
											['features_003.npy', 'scores_003.npy']
											]
							):
		self._files = files
		self._current_file = 0
		self._current_index = 0
		self._current_epoch = 0
		self._num_records_seen = 0
		self._separate_epochs = True
		self._num_minibatches = 0
		self._data_dir = utils.get_dict_value(params,'data_dir')
		self.load_next_file()
	def current_epoch(self):
		return self._current_epoch

	def current_index(self):
		return self._current_index

	def load_next_file(self):
		feature_filename = os.path.join(self._data_dir, self._files[self._current_file][0])
		scores_filename = os.path.join(self._data_dir, self._files[self._current_file][1])
		print("loading %s"%feature_filename)
		self._features = np.load(feature_filename)
		print("loading %s"%scores_filename)
		self._scores = np.load(scores_filename)

		self._current_file += 1
		self._current_file %= len(self._files)

		if self._current_file == 0:
			self._current_epoch += 1

	def next_batch(self, batch_size=2, params=None):
		features = np.empty((0, self._features.shape[1]))
		scores = np.empty((0))
		while (len(scores) < batch_size):
			end_idx = self._current_index + batch_size - len(scores)
			if end_idx > len(self._scores):
				end_idx = len(self._scores)
				next_current_idx = 0
				self.load_next_file()
			else:
				next_current_idx = end_idx
			features = np.concatenate((features, self._features[self._current_index:end_idx, :]))
			scores = np.concatenate((scores, self._scores[self._current_index:end_idx]))
			self._current_index = next_current_idx


		self._num_minibatches += 1
		return {'features': features, 'y': scores}


if __name__ == "__main__":
	d = PPTDesignerData(params={'data_dir':'./data'})
	k = d.next_batch()
	print(k)