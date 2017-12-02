import framework.utils.common as utils
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

params = utils.load_param_file(sys.argv[1])
model_name = utils.get_dict_value(params, 'model_name')
inference_output_node = utils.get_dict_value(params,'inference_output_node', 'sm_decision')
freeze_graph = os.path.join(os.path.dirname(tf.__file__), 'python/tools','freeze_graph.py')
ckpt = os.path.join(utils.get_dict_value(params, 'output_location'),
										model_name + '.ckpt')
pbf = os.path.join(utils.get_dict_value(params, 'output_location'),
									 model_name + '.training.pb')
cmd = 'python3 %s --help' % freeze_graph

cmd = 'python3 %s --input_binary True --clear_devices True --input_graph %s ' \
		'--input_checkpoint %s --output_node_names %s --output_graph %s.graphdef' % \
			(freeze_graph, pbf, ckpt, inference_output_node, model_name)
os.system(cmd)

print("WROTE %s.graphdef"%model_name)