import tensorflow as tf
import framework.subgraph.nlp as nlp
import framework.utils.common as utils
import framework.subgraph.mlp as mlp
import framework.subgraph.misc as misc
import numpy as np
import framework.subgraph.core as core

def inference(params):
	feature_count = utils.get_dict_value(params, 'feature_count')
	mlp_config = utils.get_dict_value(params, 'mlp_config')
	mlp_activations = utils.get_dict_value(params, 'mlp_activations')
	mlp_dropout_keep_probs = utils.get_dict_value(params, 'mlp_dropout_keep_probs')
	x = tf.placeholder(tf.float32, [None, feature_count], 'features')
	mlp_out, _ = mlp.fully_connected_network([x], mlp_config,
																					 layer_activations=mlp_activations,
																					 dropout_keep_probs=mlp_dropout_keep_probs)
	return mlp_out,_