import tensorflow as tf
import framework.subgraph.nlp as nlp
import framework.utils.common as utils
import framework.subgraph.mlp as mlp
import framework.subgraph.misc as misc
import framework.subgraph.core as core


def loss(network, params=None, name='rnnlm_loss'):
	logits = network[0]
	batch_size = params['batch_size']
	num_steps = params['num_steps']
	y = tf.placeholder(tf.int32, [batch_size, num_steps], name='y')
	w = tf.placeholder(tf.float32, [batch_size, num_steps], name='w')
	loss = tf.contrib.seq2seq.sequence_loss(
		logits,
		y,
		w,
		average_across_timesteps=False,
		average_across_batch=True)
	loss = tf.reduce_sum(loss)
	return [loss]

def inference(params):
	batch_size = params['batch_size']
	num_steps = params['num_steps']
	cell_size = params['cell_size']
	vocab_size = params['vocab_size']
	num_layers = params['num_layers']
	is_training = tf.get_default_graph().get_tensor_by_name('is_training:0')
	embedding_wd = utils.get_dict_value(params, 'embedding_wd')
#	embedding_device = utils.get_dict_value(params, 'embedding_device')
	embedding_initializer = utils.get_dict_value(params, 'embedding_initializer')
	embedding_keep_prob = utils.get_dict_value(params, 'embedding_keep_prob')
	rnn_dropout_keep_prob = utils.get_dict_value(params, 'rnn_dropout_keep_prob', 1.0)

	embedding_matrix = nlp.variable_with_weight_decay('embedding_matrix',
																												 [vocab_size, cell_size],
																												 initializer=embedding_initializer,
																												 wd=embedding_wd)

	words = tf.placeholder(tf.int32, [batch_size, num_steps], name='x')
	emb_words = tf.nn.embedding_lookup(embedding_matrix, words, 'emb_words')

	# add dropout if needed
	if embedding_keep_prob is not None and embedding_keep_prob < 1.0:
		[emb_words],_ = core.dropout([emb_words], [embedding_keep_prob])


	if num_layers > 1:
		cell_list = []
		for _ in range(num_layers):
			cell = tf.contrib.rnn.BasicLSTMCell(cell_size)
			if rnn_dropout_keep_prob < 1.00:
				[cell],_ = core.rnn_dropout([cell],[rnn_dropout_keep_prob])
			cell_list.append(cell)
		cell = tf.contrib.rnn.MultiRNNCell(
			cell_list, state_is_tuple=True)
	else:
		cell = tf.contrib.rnn.BasicLSTMCell(cell_size)
		if rnn_dropout_keep_prob < 1.00:
			[cell],_ = core.rnn_dropout([cell],[rnn_dropout_keep_prob])

	state = initial_state = cell.zero_state(batch_size, tf.float32)

	outputs = []
	# run through RNN
	for i in range(num_steps):
		output, state = cell(emb_words[:, i, :], state)
		outputs.append(output)

	output = tf.reshape(tf.concat(outputs,1),[-1,cell_size])
	softmax_w = nlp.variable_with_weight_decay('softmax_w',
																						 [cell_size,vocab_size])
	softmax_b = nlp.variable_with_weight_decay('softmax_b',
																						 [vocab_size])

	logits = tf.nn.xw_plus_b(output, softmax_w, softmax_b)
	logits = tf.reshape(logits, [batch_size, num_steps, vocab_size], name='output_logits')

	return [logits]