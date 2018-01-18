from framework.utils.data.text_indexer import TextIndexer
from framework.evaluator import Evaluator
import framework.utils.common as utils
from tf_model import TFModel
import os


class TFGrammarModel(TFModel):
	def __init__(self):
		None

	def load(self, model_dir):
		self._model_dir = model_dir
		self._paramsfile = os.path.join(self._model_dir, 'params.py')
		self._params = utils.load_param_file(self._paramsfile)
		ckpt = os.path.join(utils.get_dict_value(self._params,'output_location'),
												utils.get_dict_value(self._params, 'model_name') + '.ckpt')
		self._e = Evaluator.load2(ckpt)

	def get_model_name(self):
		return self._params['model_name']