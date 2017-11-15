from urllib.parse import urlparse
from http.server import BaseHTTPRequestHandler, HTTPServer
import os

class HttpHandler(BaseHTTPRequestHandler):

	def do_GET(self):
		self.send_response(200)
		self.send_header("Content-type", "text/html")
		self.end_headers()
		parsed_path = urlparse(self.path)
		if parsed_path.path == "/":
			msg = ""
			with open("index.html", "r") as f:
				for line in f:
					msg += line
		else:
				msg = "<html><body>Unhandled URL: %s?%s<br></body></html>"%(parsed_path.path, parsed_path.query)
		self.wfile.write(bytes(msg, 'utf8'))

server_port = 80
httpd = HTTPServer(("0.0.0.0", server_port), HttpHandler)
try:
	print("Starting server...")
	httpd.serve_forever()
except KeyboardInterrupt:
	pass
httpd.server_close()

