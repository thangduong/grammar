# vocab_diff <v1> <v2>
import sys

v1list = {}
with open(sys.argv[1], 'r') as f:
	for line in f:
		line = line.rstrip().lstrip().split()
		if len(line)>0:
			line = line[0]
			v1list[line]=1
v2list = {}
with open(sys.argv[2], 'r') as f:
	for line in f:
		line = line.rstrip().lstrip().split()
		if len(line)>0:
			line = line[0]
			v2list[line]=1

print("generating diff")
d1 = [x for x in v1list if x not in v2list]
d2 = [x for x in v2list if x not in v1list]

print("saving...")
with open('v1-v2.txt','w') as f:
	for d in d1:
		f.write('%s\n'% d)

with open('v2-v1.txt','w') as f:
	for d in d2:
		f.write('%s\n'% d)