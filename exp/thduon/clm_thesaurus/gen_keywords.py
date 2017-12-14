import framework.utils.common as utils
import pickle
params = utils.load_param_file('params.py')

keywords = params['keywords']
nwfl = ['cw.txt', 'cw2.txt']
for nwf in nwfl:
	with open(nwf, 'r') as f:
		for w in f:
			w = w.rstrip().lstrip().split()
			w = [x.lower() for x in w]
			if len(w) == 1:
				w = w[0]
			if w not in keywords:
				keywords.append(w)

with open('keywords.pkl', 'rb') as f:
	k2 = pickle.load(f)
	for w in k2:
		if w not in keywords:
			keywords.append(w)

print(keywords)
print(len(keywords))

with open('keywords.pkl', 'wb') as f:
	pickle.dump(keywords, f)
