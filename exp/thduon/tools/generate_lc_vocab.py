file = '/mnt/work/1-billion-word-language-modeling-benchmark/1b_word_vocab.txt'
vdict = {}
with open(file, 'r') as f:
	for line in f:
		line = line.rstrip().lstrip().split()
		word = line[0].lower()
		count = int(line[1])
		if word in vdict:
			vdict[word]+=count
		else:
			vdict[word]=count

sorted_dict = sorted(vdict.items(), key=lambda value: value[1], reverse=True)
with open('lc_vocab.txt', 'w') as f:
	for x,y in sorted_dict:
#		if y > 100:
		print('%s %s'%(x,y))
		f.write('%s %s\n'%(x,y))

