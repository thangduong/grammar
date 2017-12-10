with open('/mnt/work/1-billion-word-language-modeling-benchmark/lc_vocab.txt', 'r') as f:
	for line in f:
		line = line.rstrip().lstrip()
		pieces = line.split()
		if pieces[0].isalpha() and 'ʼ' not in pieces[0] and int(pieces[1])>200:
			print(line)
