import os
import pickle
import operator
data_dir = '/mnt/work/tokenized_training_data'
filenames = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]
vocab = {}
lowercase = True
prefix=''
if lowercase:
	prefix='lowercase_'
for i,filename in enumerate(filenames):
#	if i>2:
#		break
	print(filename)
	with open(filename,'r') as f:
		for line in f:
			line = line.rstrip().lstrip()
			if lowercase:
				line=line.lower()
			tokens = line.split()
			for token in tokens:
				if token in vocab:
					vocab[token]+=1
				else:
					vocab[token]=1


sorted_vocab = sorted(vocab.items(), key=operator.itemgetter(1), reverse=True)
with open('%svocab.pkl'%prefix,'wb') as f:
	pickle.dump(vocab, f)

with open('%svocab.txt'%prefix, 'w') as f:
	for x,y in sorted_vocab:
		print('%s %s'%(x,y))
		f.write('%s %s\n'%(x,y))

print('vocab size = %s'%len(vocab))