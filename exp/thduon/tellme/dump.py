from framework.evaluator import Evaluator
import framework.utils.common as utils
from tellme.data import TellmeData
import numpy as np
from time import time
import gflags
import os
import sys

params = utils.load_param_file('./output/tellmeV8/params.py')

ckpt = os.path.join(utils.get_dict_value(params, 'output_location'),
										utils.get_dict_value(params, 'model_name') + '.ckpt')
accuracy_file = os.path.join(utils.get_dict_value(params, 'output_location'),
														 'accuracy.txt')
e = Evaluator.load2(ckpt)
e.dump_graph()
e.save_graph_as_pbtxt('tellmev8.pbtxt')