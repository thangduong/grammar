import framework.utils.common as utils
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

paramsfile = sys.argv[1]
params = utils.load_param_file(paramsfile)
dirname = os.path.dirname(paramsfile)
model_name = utils.get_dict_value(params, 'model_name')
input_graphdef_filename = os.path.join(dirname,model_name + ".graphdef")
output_graphdef_filename = os.path.join(dirname,model_name + ".inf.graphdef")
scriptname = os.path.join(os.path.dirname(tf.__file__), 'python/tools','optimize_for_inference.py')
inference_output_node = utils.get_dict_value(params,'inference_output_node', 'sm_decision')
inference_input_nodes = utils.get_dict_value(params,'inference_input_node', 'sentence')
cmd = 'python3 %s --input=%s --output=%s --frozen_graph=True --input_names=%s --output_names=%s' % \
			(scriptname, input_graphdef_filename, output_graphdef_filename, inference_input_nodes, inference_output_node)
print(cmd)
os.system(cmd)

print("WROTE %s"%output_graphdef_filename)
