from framework.utils.data.text_indexer import TextIndexer
from framework.evaluator import Evaluator
import framework.utils.common as utils
import word_classifier.data as data
import os
import urllib.parse
import sys
from time import time
from urllib.parse import urlparse
from http.server import BaseHTTPRequestHandler, HTTPServer
run_server = True

DEFAULT_THRESHOLD = .4
paramsfile = sys.argv[1] #"params.py"
data_base_dir = ""
http_port = 8081
params = utils.load_param_file(paramsfile)

vocab_file = os.path.join(utils.get_dict_value(params,'output_location'), 'vocab.pkl')
ckpt = os.path.join(utils.get_dict_value(params,'output_location'),
										utils.get_dict_value(params, 'model_name') + '.ckpt')
print(ckpt)
e = Evaluator.load2(ckpt)
i = TextIndexer.from_file(vocab_file)

num_before = utils.get_dict_value(params, "num_words_before")
num_after = utils.get_dict_value(params, "num_words_after")
pad_tok = utils.get_dict_value(params, "pad_tok", '<pad>')

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
	msg_list = []
	for candidate_idx, candidate in enumerate(candidates):
		_, indexed, _, _ = i.index_wordlist(candidate)
		r = e.eval({'sentence': [indexed]}, {'sm_decision'})
		msg = '%s %s'%(candidate[int(len(candidate)/2)],r[0][0][1])
		msg_list.append(msg)
		print(msg)
#		print(candidate)
#		print(r)
		if r[0][0][1] > thres:
			locations.append(candidate_idx)
			probs.append(r[0][0][1])
	print(probs)
	return locations, probs, msg_list

def eval_sentence(sentence, take_top_one=False, threshold=0.5):
	sentence = sentence.replace(',', ' ,')
	sentence = sentence.replace('!', ' !')
	tokens = [x for x in sentence.split()] # if x not in keywords]
	candidates = gen_candidates(tokens, num_before, num_after, pad_tok)
	locs, probs, msg_list = eval_candidates(candidates, e, i, threshold)

	if len(locs)>0:
		probs, locs = zip(*sorted(zip(probs,locs), reverse=True))
		print(probs)
		print(locs)
		locs = locs[:1]
		probs = probs[:1]
	num_added = 0
	fixed_tok = []
	for toki, tok in enumerate(tokens):
		if toki in locs:
			fixed_tok.append('-')
			num_added += 1
		fixed_tok.append(tok)
#	print(sentence)
	fixed_sentence, idx = data.merge_tokens_for_text(fixed_tok)
#	print(fixed_sentence)
	return fixed_sentence, num_added, msg_list

class HttpHandler(BaseHTTPRequestHandler):

	def do_GET(self):
		self.send_response(200)
		parsed_path = urlparse(self.path)
		if parsed_path.path == "/decode":
			self.send_header("Content-type", "text/html")
			self.end_headers()
			sentence = urllib.parse.unquote(parsed_path.query).lower()
			done = False
			scores = []
			debug = False
			threshold = DEFAULT_THRESHOLD
			if "%%" in sentence:
				toks = sentence.split()
				for ti, tok in enumerate(toks):
					if "%%" in tok:
						print(tok)
						threshold=float(tok[2:])
						sentence = sentence.replace(tok, " ")
			if "@1" in sentence:
				sentence = sentence.replace("@1", " ")
				debug = True
			sentence = sentence.replace("'s", " 's")
			sentence = sentence.replace("n't", " n't")
			sentence = sentence.replace("\"", " \"")
			sentence = sentence.replace(".", " .")
			sentence = sentence.replace("?", " ?")
			sentence = sentence.replace("!", " !")
			sentence = sentence.replace(",", " ")
			fixed_sentence = sentence
			while not done:
				fixed_sentence, num_added, msg_list = eval_sentence(fixed_sentence,threshold=threshold)
				scores += msg_list + ["----------"]
				print("num added: %s"%num_added)
				done = num_added == 0
			msg = fixed_sentence
			if debug:
				msg = msg + "<br><br>" + "<br>".join(scores)
			self.wfile.write(bytes(msg, 'utf8'))
		elif parsed_path.path == "/":
			self.send_header("Content-type", "text/html")
			self.end_headers()
			msg = ""
			with open("index.html", "r") as f:
				for line in f:
					msg += line.replace("___MODEL_NAME___", self.model_name)
			self.wfile.write(bytes(msg, 'utf8'))
		elif parsed_path.path.endswith(".png"):

			self.send_header("Content-type", "image/png")
			self.end_headers()
			with open("."+parsed_path.path, "rb") as f:
				self.wfile.write(f.read())
		else:
			self.send_header("Content-type", "text/html")
			self.end_headers()
			msg = "<html><body>Unhandled URL: %s?%s<br></body></html>"%(parsed_path.path, parsed_path.query)
			self.wfile.write(bytes(msg, 'utf8'))

if run_server:
	HttpHandler.model_name = utils.get_dict_value(params, 'model_name', '_UNKNOWN_MODEL_')
	httpd = HTTPServer(("0.0.0.0", http_port), HttpHandler)
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


