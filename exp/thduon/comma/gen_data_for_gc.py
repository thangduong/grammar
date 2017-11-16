import random
from time import time
import word_classifier.data as data
import pickle

session = int(time())

def _gen_candidates(tokens, error_list = []):
	result = [[0, tokens, error_list]]
	if ',' in tokens:
		for toki, tok in enumerate(tokens):
			if tok == ',':
				ms = tokens[:toki] + tokens[toki+1:]
				result.append([0, ms, error_list+[[-1, toki]]])

	# number of commas added incorrectly
	n_added = len(result)
	for n in range(n_added):
		ip = random.randint(0, len(tokens))
		ms = tokens[:ip] + [','] + tokens[ip:]
		result.append([0, ms, error_list+[[1, ip]]])
	return result

def gen_candidates(tokens, num_times = 1):
	results = [[0, tokens, []]]
	for times in range(num_times):
		last_results = results
		results = []
		for r in last_results:
			results += _gen_candidates(r[1], r[2])
	return results
random.seed(time())

def add_index(cands):
	for candi, cand in enumerate(cands):
		cand[0] = candi

filenames = [
	'/mnt/work/1-billion-word-language-modeling-benchmark/heldout-monolingual.tokenized.shuffled/news.en.heldout-00000-of-00050'
	,'/mnt/work/1-billion-word-language-modeling-benchmark/heldout-monolingual.tokenized.shuffled/news.en.heldout-00001-of-00050'
	,'/mnt/work/1-billion-word-language-modeling-benchmark/heldout-monolingual.tokenized.shuffled/news.en.heldout-00002-of-00050'
	,'/mnt/work/1-billion-word-language-modeling-benchmark/heldout-monolingual.tokenized.shuffled/news.en.heldout-00003-of-00050'
	,'/mnt/work/1-billion-word-language-modeling-benchmark/heldout-monolingual.tokenized.shuffled/news.en.heldout-00004-of-00050'
	,'/mnt/work/1-billion-word-language-modeling-benchmark/heldout-monolingual.tokenized.shuffled/news.en.heldout-00005-of-00050'
	,'/mnt/work/1-billion-word-language-modeling-benchmark/heldout-monolingual.tokenized.shuffled/news.en.heldout-00006-of-00050'
	,'/mnt/work/1-billion-word-language-modeling-benchmark/heldout-monolingual.tokenized.shuffled/news.en.heldout-00007-of-00050'
	,'/mnt/work/1-billion-word-language-modeling-benchmark/heldout-monolingual.tokenized.shuffled/news.en.heldout-00008-of-00050'
	,'/mnt/work/1-billion-word-language-modeling-benchmark/heldout-monolingual.tokenized.shuffled/news.en.heldout-00009-of-00050'
	,'/mnt/work/1-billion-word-language-modeling-benchmark/heldout-monolingual.tokenized.shuffled/news.en.heldout-00010-of-00050'
	]
	#cand = gen_candidates("the quick brown fox jumped over , the lazy dog".split(), 2)
all_cand = []
for filename in filenames:
	with open(filename) as f:
		for sentence in f:
			cand = gen_candidates(sentence.split(),2)
			all_cand += cand

random.shuffle(all_cand)
add_index(all_cand)

with open('gc_comma_data_%s.txt'%session,'w') as f:
	for c in all_cand:
		sentence, loc = data.merge_tokens_for_text(c[1])
		c.append([sentence, loc])
		f.write('%d\t%s\n'%(c[0],sentence))
		print('%d\t%s'%(c[0],sentence))
with open('gc_comma_data_%s.pkl'%session, 'wb') as f:
	pickle.dump(all_cand, f)

