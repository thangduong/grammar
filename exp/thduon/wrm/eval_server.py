from framework.utils.data.text_indexer import TextIndexer
from framework.evaluator import Evaluator
import framework.utils.common as utils
import word_classifier.data as data
import os
import urllib.parse
from time import time
from urllib.parse import urlparse
from http.server import BaseHTTPRequestHandler, HTTPServer
from wrmeval import WRMEval

http_port = 8080
e = WRMEval()
e.load('./output/wrmV3/')

class HttpHandler(BaseHTTPRequestHandler):

	def do_GET(self):
		self.send_response(200)
		self.send_header("Content-type", "text/html")
		self.end_headers()
		parsed_path = urlparse(self.path)
		if parsed_path.path == "/decode":
			sentence = urllib.parse.unquote(parsed_path.query)
			sentence = sentence.replace("'s", " 's")
			sentence = sentence.replace("\"", " \"")
			sentence = sentence.replace(".", " .")
			sentence = sentence.replace("?", " ?")
			sentence = sentence.replace("!", " !")
			corrections, tokens = e.critique(sentence)
			markup = e.markup_critique(corrections, tokens)
			msg = markup
		elif parsed_path.path == "/":
			msg = ""
			with open("index.html", "r") as f:
				for line in f:
					msg += line.replace("___MODEL_NAME___", self.model_name)
		else:
				msg = "<html><body>Unhandled URL: %s?%s<br></body></html>"%(parsed_path.path, parsed_path.query)
		self.wfile.write(bytes(msg, 'utf8'))

HttpHandler.model_name = e.get_model_name()
httpd = HTTPServer(("0.0.0.0", http_port), HttpHandler)
try:
	print("Starting server...")
	httpd.serve_forever()
except KeyboardInterrupt:
	pass
httpd.server_close()
