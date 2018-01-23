import sys
sys.path.append('./dlframework')
from framework.evaluator import Evaluator
import framework.utils.common as utils
import tensorflow as tf
import json
import csv
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('input_datafile', 'fakedata.tsv', 'input data tsv file')
tf.app.flags.DEFINE_string('output_prfile', 'pr.txt', 'output pr file')
tf.app.flags.DEFINE_string('model_dir', 'sapV6', 'output pr file')



with open(os.path.join(FLAGS.model_dir, 'params.json'),'r') as f:
	params = json.load(f)

#params = utils.load_param_file(paramsfile)
gd = os.path.join(FLAGS.model_dir, utils.get_dict_value(params, 'model_name') + '.graphdef')

e = Evaluator.load_graphdef(gd)

punctuations = params['punctuations']			# false negative

num_chars_before = params['num_chars_before']
num_chars_after = params['num_chars_after']
vocab_size = params['vocab_size']

unhandled_count = 0

model_result = []
ground_truth = []
with open(FLAGS.input_datafile, 'r', encoding='utf-8', errors="ignore") as f:
	tsvin = csv.reader(f, delimiter='\t')
	for row in tsvin:
		if (len(row)>10):
			behavior = row[1]
			if behavior == 'Office.NaturalLanguage.Proofing.Behaviors.Grammar.Ignore' or \
					behavior == 'Office.NaturalLanguage.Proofing.Behaviors.Grammar.Ignore':
				ground_truth.append(1)
			else:
				ground_truth.append(0)
			sentence = row[8]
			critique_start = int(row[9])
			critique_len = int(row[10])
			i = critique_start
			sentence_ord = [ord(x) for x in sentence]
			filter_prob = -1
			while i < critique_start + critique_len:
				if sentence[i] in punctuations:
					before = [0] * num_chars_before + sentence_ord[:i+1]
					after = sentence_ord[i+1:] + [0] * num_chars_after
					before = before[-num_chars_before:]
					after = after[:num_chars_after]
					input = before + after
					input_str = [chr(x) for x in input]
					result = e.eval({'sentence': [input]}, ['sm_decision'])
					filter_prob = result[0][0][0]
					break
				i += 1
			if filter_prob < 0:
				unhandled_count += 1
				filter_prob = 0
			model_result.append(filter_prob)
#			print("%s %s"%(ground_truth[-1], filter_prob))


with open(FLAGS.output_prfile, 'w') as prfile:
	prfile.write("Threshold\tPrecision\tRecall\tFPR\r\n")
	for i in range(0,101):
		thres = i / 100
		tp = fp = fn = tn = 0
		for model, gold in zip(model_result, ground_truth):
			if model >= thres and gold == 1:
				# true positive
				tp += 1
			elif model >= thres and gold == 0:
				# false positive
				fp += 1
			elif model < thres and gold == 1:
				# false negative
				fn += 1
			elif model < thres and gold == 0:
				# true negative
				tn += 1
#		print("%s %s %s %s %s"%(thres,tp,fp,fn,tn))
		p = tp / max((tp + fp),1)
		r = tp / max((tp + fn),1)
		prfile.write("%0.4f\t%0.4f\t%0.4f\t%0.4f\r\n"%(thres, p, r, fp/max((fp+tn),1)))