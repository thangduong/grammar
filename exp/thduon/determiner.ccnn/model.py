import tensorflow as tf
import framework.subgraph.nlp as nlp
import framework.utils.common as utils
import framework.subgraph.mlp as mlp
import framework.subgraph.misc as misc
import framework.subgraph.core as core

#def word_encoder(word_emb, params, name='encoded_word'):


def sentence_encoder(emb_sentence, word_emb, params, name='encoded_sentence'):
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

	char_conv_num_features = utils.get_dict_value(params, 'char_conv_num_features')
	char_conv_widths = utils.get_dict_value(params, 'char_conv_widths')
	char_use_no_conv_path = utils.get_dict_value(params, 'char_use_no_conv_path', False)

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

	# deal with the word path
	if char_conv_widths is not None and char_conv_num_features is not None:
		if char_use_no_conv_path:
			conv_group.append(word_emb)
		for i, (char_conv_num_feature, char_conv_width) in enumerate(zip(char_conv_num_features, char_conv_widths)):
			conv_out = nlp.conv1d_array(word_emb, char_conv_num_feature, char_conv_width,name='conv%s'%(str(i)), w_wds=0.000, b_wds=0.000, keep_probs=conv_keep_probs)
			conv_group.append(conv_out)
	else:
		conv_group.append(word_emb)

	conv_out, _ = misc.concat(conv_group)
	mlp_out, _ = mlp.fully_connected_network(conv_out, mlp_config, layer_activations=mlp_activations, dropout_keep_probs=mlp_dropout_keep_probs)
	return [tf.identity(mlp_out[0], name=name)], {}


def inference(params):
	embedding_size = params['embedding_size']
	vocab_size = params['vocab_size']
	sentence_len = params['num_words_before'] + params['num_words_after']
	embedding_wd = utils.get_dict_value(params, 'embedding_wd')
	embedding_device = utils.get_dict_value(params, 'embedding_device')
	embedding_initializer = utils.get_dict_value(params, 'embedding_initializer')
	embedding_keep_prob = utils.get_dict_value(params, 'embedding_keep_prob')
	word_embedding_size = utils.get_dict_value(params, 'word_embedding_size', embedding_size)
	word_len = params['word_len']

	print("USING EMBEDDING DEVICE %s" %embedding_device)
	if embedding_device is not None:
		with tf.device(embedding_device):
			embedding_matrix = nlp.variable_with_weight_decay('embedding_matrix',
																												[vocab_size, embedding_size],
																												initializer=embedding_initializer, wd=embedding_wd)
	else:
		embedding_matrix = nlp.variable_with_weight_decay('embedding_matrix', [vocab_size, embedding_size],
																											initializer=embedding_initializer, wd=embedding_wd)
	word_embedding_matrix = nlp.variable_with_weight_decay('word_embedding_matrix', [128, word_embedding_size],
																										initializer=embedding_initializer, wd=embedding_wd)

	input_sentence = tf.placeholder(tf.int32, [None, sentence_len], 'sentence')
	input_word = tf.placeholder(tf.int32, [None, word_len], 'word')
	emb_sentence = tf.nn.embedding_lookup(embedding_matrix, input_sentence, 'emb_sentence')
	if embedding_keep_prob is not None and embedding_keep_prob < 1.0:
		[emb_sentence],_ = core.dropout([emb_sentence], [embedding_keep_prob])
	emb_word = tf.nn.embedding_lookup(word_embedding_matrix, input_word, 'emb_word')
	enc_sentence, _ = sentence_encoder(emb_sentence, emb_word, params)

	return enc_sentence, None
