try:
	from ae365_utils.comp_logger import *
	def print_log(msg):
		log(x)
	print_log("Running in AE365")
except:
	def print_log(msg):
		print(msg)
	print_log("Not running in AE365")

debug = True
if debug:
	print_log("Importing sys")
import sys
sys.path.append('./dlframework')
if debug:
	print_log("Importing framework.evaluator")
from framework.evaluator import Evaluator
if debug:
	print_log("Importing framework.utils.common")
import framework.utils.common as utils
if debug:
	print_log("Importing tensorflow")
import tensorflow as tf
if debug:
	print_log("Importing json")
import json
if debug:
	print_log("Importing csv")
import csv
if debug:
	print_log("Importing os")
import os
if debug:
	print_log("Importing os")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

if debug:
	print_log("Defining flags")
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('input_datafile', 'fakedata.tsv', 'input data tsv file')
tf.app.flags.DEFINE_string('output_prfile', 'pr.txt', 'output pr file')
tf.app.flags.DEFINE_string('model_dir', 'sapV6', 'output pr file')

def commonprefix(m):
	# got this from stackoverflow!
	if not m: return ''
	s1 = min(m)
	s2 = max(m)
	for i, c in enumerate(s1):
		if c != s2[i]:
			return s1[:i]
	return s1

paramsfile = os.path.join(FLAGS.model_dir, 'params.json')

if debug:
	print_log("Loading %s"%paramsfile)

with open(paramsfile,'r') as f:
	params = json.load(f)

if debug:
	print_log("Done loading %s"%paramsfile)
gd = os.path.join(FLAGS.model_dir, utils.get_dict_value(params, 'model_name') + '.graphdef')

if debug:
	print_log("Loading %s"%gd)
e = Evaluator.load_graphdef(gd)
if debug:
	print_log("Done loading %s"%gd)

punctuations = params['punctuations']			# false negative

num_chars_before = params['num_chars_before']
num_chars_after = params['num_chars_after']
vocab_size = params['vocab_size']

unhandled_count = 0

model_result = []
ground_truth = []
if debug:
	print_log("Processing %s"%FLAGS.input_datafile)
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

#			for a,x in enumerate(row):
#				print_log("%s: %s"%(a,x))
			flagged_text = row[6]
			suggestion1 = row[13]
#			print_log([flagged_text, suggestion1])
			pref = commonprefix([flagged_text, suggestion1])
#			print_log(pref)

			punctuation_idx = len(pref)-1 + critique_start

#			print_log("PUNCTUATION %s"%sentence[punctuation_idx])
			#while i < critique_start + critique_len:
			if sentence[punctuation_idx] in punctuations:
				before = [0] * num_chars_before + sentence_ord[:punctuation_idx+1]
				after = sentence_ord[punctuation_idx+1:] + [0] * num_chars_after
				before = before[-num_chars_before:]
				after = after[:num_chars_after]
				input = before + after
				input_str = [chr(x) for x in input]
				result = e.eval({'sentence': [input]}, ['sm_decision'])
				filter_prob = result[0][0][0]
			#	i += 1
			if filter_prob < 0:
				unhandled_count += 1
				filter_prob = 0
			model_result.append(filter_prob)
#			print_log("%s %s"%(ground_truth[-1], filter_prob))

if debug:
	print_log("Writing %s"%FLAGS.output_prfile)

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
#		print_log("%s %s %s %s %s"%(thres,tp,fp,fn,tn))
		p = tp / max((tp + fp),1)
		r = tp / max((tp + fn),1)
		prfile.write("%0.4f\t%0.4f\t%0.4f\t%0.4f\r\n"%(thres, p, r, fp/max((fp+tn),1)))

print_log("Unhandled count = %s"%unhandled_count)
if debug:
	print_log("All done.  Exiting.")

