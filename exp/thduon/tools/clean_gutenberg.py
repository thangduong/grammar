import json
import os
from tokenex.tokenizer import Tokenizer
import sys

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

data_dir = '/data/gutenberg_data/Gutenberg/txt/'
fn_file = '/data/gutenberg_data/Gutenberg/files.txt'
with open(fn_file, 'r', encoding='utf-8', errors='ignore') as f:
	files = [line.rstrip().lstrip() for line in f]

#filenames = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]
filenames = [os.path.join(data_dir, f) for f in files]


tok = Tokenizer()
def split_lines(tok, para):
#	print("\n\n")
#	print(para)
#	print("\n\n")
	tokens, starts, lens, _ = tok.tokenize_ex2(para)
#	print(tokens)
#	print(starts)
#	print(lens)
	line_splitters = ['.','?','!']
	lines = []
	last_pos = 0
	data_iter = enumerate(zip(tokens, starts, lens))
	quote_count = 0
	line_prefix = ""
	for i, (token, start, lenv) in data_iter:
		if token == '"':
			quote_count += 1
		if token in line_splitters:
			endpos = start+lenv
			if i < len(tokens)-1:
				if tokens[i+1] == "\"" and ((quote_count%2) == 1):
					quote_count += 1
					endpos = starts[i+1] + lens[i+1]
					if i < len(starts) - 2:
						next_last_pos = starts[i + 2]
					else:
						next_last_pos = len(para)
					next(data_iter)
				else:
					if i < len(starts)-1:
						next_last_pos = starts[i+1]
					else:
						next_last_pos = len(para)
			else:
				next_last_pos = len(para)
			line = line_prefix+para[last_pos:endpos]
			if ((quote_count%2) == 1):
				line += "\""
				line_prefix = "\""
				quote_count = 1
			else:
				line_prefix = ""
				quote_count = 0
			print(line)
			lines.append(line)
			last_pos = next_last_pos
	if last_pos < len(para):
		line = para[last_pos:]
		lines.append(line)
	return lines

#x= "CROME YELLOW By Aldous Huxley CHAPTER I. Along this particular stretch of line no express had ever passed. All the trains--the few that there were--stopped at all the stations. Denis knew the names of those stations by heart. Bole, Tritton, Spavin Delawarr, Knipswich for Timpany, West Bowlby, and, finally, Camlet-on-the-Water. Camlet was where he always got out, leaving the train to creep indolently onward, goodness only knew whither, into the green heart of England. They were snorting out of West Bowlby now. It was the next station, thank Heaven. Denis took his chattels off the rack and piled them neatly in the corner opposite his own. A futile proceeding. But one must have something to do. When he had finished, he sank back into his seat and closed his eyes. It was extremely hot. Oh, this journey! It was two hours cut clean out of his life; two hours in which he might have done so much, so much--written the perfect poem, for example, or read the one illuminating book. Instead of which--his gorge rose at the smell of the dusty cushions against which he was leaning. Two hours. One hundred and twenty minutes. Anything might be done in that time. Anything. Nothing. Oh, he had had hundreds of hours, and what had he done with them? Wasted them, spilt the precious minutes as though his reservoir were inexhaustible. Denis groaned in the spirit, condemned himself utterly with all his works. What right had he to sit in the sunshine, to occupy corner seats in third-class carriages, to be alive? None, none, none. Misery and a nameless nostalgic distress possessed him. He was twenty-three, and oh! so agonizingly conscious of the fact. The train came bumpingly to a halt. Here was Camlet at last. Denis jumped up, crammed his hat over his eyes, deranged his pile of baggage, leaned out of the window and shouted for a porter, seized a bag in either hand, and had to put them down again in order to open the door. When at last he had safely bundled himself and his baggage on to the platform, he ran up the train towards the van. \"A bicycle, a bicycle!\" he said breathlessly to the guard. He felt himself a man of action. The guard paid no attention, but continued methodically to hand out, one by one, the packages labelled to Camlet. \"A bicycle!\" Denis repeated. \"A green machine, cross-framed, name of Stone. S-T-O-N-E.\" \"All in good time, sir,\" said the guard soothingly. He was a large, stately man with a naval beard. One pictured him at home, drinking tea, surrounded by a numerous family. It was in that tone that he must have spoken to his children when they were tiresome. \"All in good time, sir.\" Denis's man of action collapsed, punctured."

#split_lines(tok, x)
#exit(0)

for filename in filenames:
	sys.stderr.write("%s\n" % filename)
	line_splitters = ['.', '?', '!']
	with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
		sentence_fragment = ""
		para = ""
		bad_para = False
		for line in f:
			line = line.rstrip().lstrip()
			if len(line) == 0:
				if not bad_para:
					slines = split_lines(tok, para)
					for sline in slines:
						print(sline)
				para = ""
				bad_para = False
			else:
				if line[0] == ' ':
					bad_para = True
				para += " " + line