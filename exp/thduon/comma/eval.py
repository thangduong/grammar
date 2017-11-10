from framework.utils.data.text_indexer import TextIndexer
from framework.evaluator import Evaluator
import framework.utils.common as utils
import word_classifier.data as data
import os
from urllib.parse import urlparse
from http.server import BaseHTTPRequestHandler, HTTPServer
run_server = False

def gen_candidates(tokens, num_before, num_after, pad_tok):
	num_tokens = len(tokens)
	tokens = [pad_tok] * num_before + tokens + [pad_tok]*num_after
	n0 = []
	n1 = []
	for toki in range(0, num_tokens):
			n0.append(tokens[toki:(toki + num_before + num_after)])
	return n0

def eval_candidates(candidates, evaluator, indexer, thres):
	locations = []
	probs = []
	for candidate_idx, candidate in enumerate(candidates):
		_, indexed, _, _ = i.index_wordlist(candidate)
		r = e.eval({'sentence': [indexed]}, {'sm_decision'})
		print('%s %s'%(candidate[int(len(candidate)/2)],r[0][0][1]))
#		print(candidate)
#		print(r)
		if r[0][0][1] > thres:
			locations.append(candidate_idx)
			probs.append(r[0][0][1])
	print(probs)
	return locations


params = utils.load_param_file('params.10_10.py')

vocab_file = os.path.join(utils.get_dict_value(params,'output_location'), 'vocab.pkl')
ckpt = os.path.join(utils.get_dict_value(params,'output_location'),
										utils.get_dict_value(params, 'model_name') + '.ckpt')

sentences = ['To evaluate if the Transformer can generalize to other tasks we performed experiments on English constituency parsing',
						 'We also trained it in a semi-supervised setting , using the larger high-confidence and BerkleyParser corpora from with approximately 17M sentences',
						 'We used a vocabulary of 16K tokens for the WSJ only setting and a vocabulary of 32K tokens for the semi-supervised setting',
						 'Our results in Table 4 show that despite the lack of task-specific tuning our model performs surprisingly well , yielding '
						  'better results than all previously reported models with the exception of the Recurrrent Neural Network Grammar',
						 'In this work we presented the Transformer , the first sequence transduction model based entirely on'
						  ' attention replacing the recurrent layers most commonly used in encoder-decoder architectures with multi-headed self-attention',

						 'We are excited about the future of attention-based models and plan to apply them to other tasks',
						 'We plan , to extend the Transformer , to problems involving input and output modalities other than text and '
						  'to investigate local restricted attention mechanisms to efficiently handle large inputs and outputs '
							'such as images audio and video',
						'Of all the tasks that are deep learning is the most difficult'
						 ]
sentences = ['The apple , which is rotten is not edible']

e = Evaluator.load2(ckpt)
i = TextIndexer.from_file(vocab_file)

num_before = utils.get_dict_value(params, "num_words_before")
num_after = utils.get_dict_value(params, "num_words_after")
pad_tok = utils.get_dict_value(params, "pad_tok", '<pad>')

keywords =  [',']
for sentence in sentences:
	tokens = [x for x in sentence.split()] # if x not in keywords]
	candidates = gen_candidates(tokens, num_before, num_after, pad_tok)
	locs = eval_candidates(candidates, e, i, 0.5)

	fixed_tok = []
#	print(locs)
	for toki, tok in enumerate(tokens):
		if toki in locs:
			fixed_tok.append(',')
		fixed_tok.append(tok)
	print(sentence)
	fixed_sentence, idx = data.merge_tokens_for_text(fixed_tok)
	print(fixed_sentence)


keywords = [',']
class HttpHandler(BaseHTTPRequestHandler):

	def do_GET(self):
		self.send_response(200)
		self.send_header("Content-type", "text/html")
		self.end_headers()
		parsed_path = urlparse(self.path)
		if parsed_path.path == "/decode":
			sentence = parsed_path.query.replace("%20", " ")
			sentence = sentence.replace("%22", "\"")
			sentence = sentence.replace(",", " , ")
			sentence = sentence.replace(".", " . ")
			print(ord(sentence[0]))

			tokens = [x for x in sentence.split() if x not in keywords]
			print(tokens)
			candidates = gen_candidates(tokens, num_before, num_after, pad_tok)
			locs = eval_candidates(candidates, e, i, 0.5)

			fixed_tok = []
			#	print(locs)
			for toki, tok in enumerate(tokens):
				if toki in locs:
					fixed_tok.append(',')
				fixed_tok.append(tok)
			print(sentence)
			fixed_sentence, idx = data.merge_tokens_for_text(fixed_tok)
			print(fixed_sentence)
			msg = "%s"%fixed_sentence
		elif parsed_path.path == "/":
			msg = ""
			with open("index.html", "r") as f:
				for line in f:
					msg += line.replace("___MODEL_NAME___", self.model_name)
		else:
				msg = "<html><body>Unhandled URL: %s?%s<br></body></html>"%(parsed_path.path, parsed_path.query)
		self.wfile.write(bytes(msg, 'utf8'))

if run_server:
	HttpHandler.model_name = utils.get_dict_value(params, 'model_name', '_UNKNOWN_MODEL_')
	httpd = HTTPServer(("0.0.0.0", 8080), HttpHandler)
	try:
		print("Starting server...")
		httpd.serve_forever()
	except KeyboardInterrupt:
		pass
	httpd.server_close()

#	for j in range(5):
#		sentence = '<pad> ' + sentence + " <pad>"
#	_,indexed,_,_ = i.index_wordlist(sentence.split())
#	r = e.eval({'sentence': [indexed]}, {'sm_decision'})
#	print(sentence)
#	print(r[0][0][1])


