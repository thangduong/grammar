import tensorflow as tf
import framework.subgraph.nlp as nlp
import framework.utils.common as utils
import framework.subgraph.mlp as mlp
import framework.subgraph.misc as misc
import framework.subgraph.core as core

def encoder(emb_sentence, params, name='encoded'):
	"""
	@param emb_sentence:
	@param params:
	@return:
	"""
	conv_num_features = utils.get_dict_value(params, 'conv_num_features')
	conv_widths = utils.get_dict_value(params, 'conv_widths')
	conv_keep_probs = utils.get_dict_value(params, 'conv_keep_probs')
	mlp_config = utils.get_dict_value(params, 'mlp_config')
	bipass_conv = utils.get_dict_value(params, 'bipass_conv')
	mlp_activations = utils.get_dict_value(params, 'mlp_activations')
	mlp_dropout_keep_probs = utils.get_dict_value(params, 'mlp_keep_probs')
	use_no_conv_path = utils.get_dict_value(params, 'use_no_conv_path')
	if bipass_conv:
		conv_group = [emb_sentence]
	else:
		if use_no_conv_path:
			conv_group = [emb_sentence]
		else:
			conv_group = []
		for i, (conv_num_feature, conv_width) in enumerate(zip(conv_num_features, conv_widths)):
			conv_out = nlp.conv1d_array(emb_sentence, conv_num_feature, conv_width,name='conv%s'%(str(i)), w_wds=0.000, b_wds=0.000, keep_probs=conv_keep_probs)
			conv_group.append(conv_out)
	conv_out, _ = misc.concat(conv_group)
	mlp_out, _ = mlp.fully_connected_network(conv_out, mlp_config, layer_activations=mlp_activations, dropout_keep_probs=mlp_dropout_keep_probs)
	return [tf.identity(mlp_out[0], name=name)], {}


def inference(params):
	embedding_size = params['embedding_size']
	sentence_len = params['num_before'] + params['num_after']
	embedding_wd = utils.get_dict_value(params, 'embedding_wd')
	embedding_device = utils.get_dict_value(params, 'embedding_device')
	embedding_initializer = utils.get_dict_value(params, 'embedding_initializer')
	embedding_keep_prob = utils.get_dict_value(params, 'embedding_keep_prob')
	word_embedding_size = utils.get_dict_value(params, 'word_embedding_size', embedding_size)

	if embedding_device is not None:
		with tf.device(embedding_device):
			word_embedding_matrix = nlp.variable_with_weight_decay('word_embedding_matrix', [256, word_embedding_size],
																														 initializer=embedding_initializer, wd=embedding_wd)
	else:
		word_embedding_matrix = nlp.variable_with_weight_decay('word_embedding_matrix', [256, word_embedding_size],
																													 initializer=embedding_initializer, wd=embedding_wd)

	if embedding_keep_prob is not None and embedding_keep_prob < 1.0:
		[word_embedding_matrix],_ = core.dropout([word_embedding_matrix], [embedding_keep_prob])
	input_sentence = tf.placeholder(tf.int32, [None, sentence_len], 'sentence')
	emb_sentence = tf.nn.embedding_lookup(word_embedding_matrix, input_sentence, 'emb_word')
	enc_sentence, _ = encoder(emb_sentence, params)

	return enc_sentence, None