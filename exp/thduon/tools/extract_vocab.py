import os
import pickle
import operator
import gflags
data_dir = '/mnt/work/test' #/mnt/work/tokenized_training_data'
data_dir = '/mnt/work/training_data.tok'
filenames = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]
vocab = {}
lowercase = True
prefix='test_'
if lowercase:
	prefix='lowercase_'
for i,filename in enumerate(filenames):
#	if i>2:
#		break
	if filename.endswith('.idea'):
		continue
	print(filename)
	with open(filename,'r',encoding='utf-8',errors='ignore') as f:
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
