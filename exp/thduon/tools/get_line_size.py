file = '/mnt/work/training_data/b_3_out_4.txt'
ll = []
with open(file,'r') as f:
	for line in f:
		line = line.rstrip().lstrip()
		ll += [len(line)]
for l in ll:
	print(l)