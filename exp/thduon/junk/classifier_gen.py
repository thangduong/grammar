

def gen_data(tokens, keywords, num_before=5, num_after=5, pad_tok="<pad>"):
	tokens = [pad_tok] * num_before + tokens + [pad_tok]*num_after
	for toki, tok in enumerate(tokens):
		if tok in keywords:
			idx = keywords.index(tok)
			before_idx = toki - num_before
			after_idx = toki + num_after + 1
			yield [tokens[before_idx:toki] + tokens[toki+1:after_idx], idx + 1]
		elif toki > num_before and toki < len(tokens)-num_after:
			before_idx = toki - num_before
			after_idx = toki + num_after + 1
			yield [tokens[before_idx:toki] + tokens[toki+1:after_idx], 0]

def gen_data_from_file(filename, keywords=[','], num_before=5, num_after=5, pad_tok="<pad>"):
	with open(filename) as f:
		for line in f:
			line = line.rstrip().lstrip()
			tokens = line.split()
			yield from gen_data(tokens, keywords=keywords, num_before=num_before, num_after=num_after, pad_tok=pad_tok)

result = gen_data_from_file('/mnt/work/1-billion-word-language-modeling-benchmark/training-monolingual.tokenized.shuffled/news.en-00001-of-00100')
for x in result:
	print(x)
