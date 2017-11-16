from framework.utils.data.text_indexer import TextIndexer
from framework.evaluator import Evaluator
import framework.utils.common as utils
import word_classifier.data as data
import os
import urllib.parse
from time import time
from urllib.parse import urlparse
from http.server import BaseHTTPRequestHandler, HTTPServer
run_server = True


paramsfile = "params.py"
data_base_dir = ""
http_port = 8080
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


def split_sentence_for_eval(sentence, keywords, num_before, num_after):
	result = data.gen_data(sentence, keywords, num_before=num_before, num_after=num_after,
                          ignore_negative_data=True, add_redundant_keyword_data=False)
	return result

def merge_sentence(sentence, num_before, num_after, mid_word):
	before = sentence[:num_before]
	after = sentence[num_after:]
	rlist = (before + [mid_word] + after)
	rlist = [x for x in rlist if x!="<pad>"]
	result = ' '.join(rlist)
	return result

def score_text(text):
	result = list(split_sentence_for_eval(text.split(), ["___"], num_before, num_after))
	filled_list = []
	for sentence, score in result:
		sentence = result[0][0]
		isentence = i.index_wordlist(sentence)
		btime = time()
		x = e.eval({'sentence':[isentence[1]]},{'sm_decision'})
		atime = time()
		pr = x[0][0][1]
		if pr > .5:
			mid_word = "whom"
		else:
			mid_word = "who"
		filled = (merge_sentence(sentence, num_before, num_after, "<u>"+mid_word+"</u>"))
		filled_list.append((filled, pr, x[0][0], atime-btime))
	return filled_list

class HttpHandler(BaseHTTPRequestHandler):

	def do_GET(self):
		self.send_response(200)
		self.send_header("Content-type", "text/html")
		self.end_headers()
		parsed_path = urlparse(self.path)
		if parsed_path.path == "/decode":
			sentence = urllib.parse.unquote(parsed_path.query)
			sentence = sentence.replace("'s", " ")
			sentence = sentence.replace("\"", " \"")
			sentence = sentence.replace(".", " .")
			sentence = sentence.replace("?", " ?")
			sentence = sentence.replace("!", " !")
			if "___" not in sentence:
				msg = "\"%s\" does not contain a blank (___ = three underscores)"%sentence
			else:
				scored_text = score_text(sentence)
				msg = ""
				for fixed_sentence, pr, sm, etime in scored_text:
					msg += "%s %s: %s"%(sm, pr,fixed_sentence)
					msg += "<br> Total runtime = %s"%(etime)
					msg += "<br>"
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


