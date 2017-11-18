# this python file generates the vocab from en.cmt file from mike
# the file first needs to be converted to text

fn = '/data/nlx_data/from_mike/en.cmt.txt'
vocab = {}
with open(fn, 'r', errors='ignore') as f:
	for line in f:
		line = line.rstrip().lstrip()
		cols = line.split('\t')
		print(cols[1])
		if cols[1] in vocab:
			vocab[cols[1]] += 1
		else:
			vocab[cols[1]] = 1

print(vocab)
print(len(vocab))