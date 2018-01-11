import json
from tokenex.tokenizer import Tokenizer
filename = '/mnt/work/reddit_data/RC_2015-01'


def split_lines(tok, para):
	tokens, starts, lens, _ = tok.tokenize_ex2(para)
#	print(tokens)
#	print(starts)
#	print(lens)
	line_splitters = ['.','?','!']
	lines = []
	last_pos = 0
	data_iter = enumerate(zip(tokens, starts, lens))
	for i, (token, start, lenv) in data_iter:
		if token in line_splitters:
			line = para[last_pos:(start+lenv)]
			lines.append(line)
			if i < len(starts)-1:
				last_pos = starts[i+1]
			else:
				last_pos = len(para)
	if last_pos < len(para):
		line = para[last_pos:]
		lines.append(line)
	return lines

tok = Tokenizer()

with open(filename, 'r') as f:
	for line in f:
		line = line.rstrip().lstrip()
		line_json = json.loads(line)
		body = line_json['body'].rstrip().lstrip()
		body = body.replace('\r', ' ')
		body = body.replace('\n', ' ')
		body = body.replace('  ', ' ')
		if len(body) > 0 and not(body == '[deleted]'):
			lines = split_lines(tok, body)
			for _line in lines:
				print("%s"%_line)
