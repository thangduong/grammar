# vocab_diff <v1> <v2>
import sys

min_freq_union = 5
min_freq = 1000

v2list = {}
with open(sys.argv[2], 'r') as f:
	for line in f:
		line = line.rstrip().lstrip().split()
		if len(line)>0:
			v2list[line[0]]=line[1]

with open(sys.argv[1], 'r') as f:
	for line in f:
		line = line.rstrip().lstrip().split()
		if len(line)>0:
			freq = int(line[1])
			if freq > min_freq or ((line[0] in v2list) and freq > min_freq_union):
				print("%s %s" % (line[0],freq))
