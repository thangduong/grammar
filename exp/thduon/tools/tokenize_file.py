from nlp.tokenizer import tokenize
import operator

filename = '/mnt/work/1-billion-word-language-modeling-benchmark/training-monolingual/news.2007.en.shuffled'

token_dict = {}
with open(filename) as f:
	for idx, line in enumerate(f):
		tokens, token_positions = tokenize(line, verbose=False)
		for token in tokens:
			if token in token_dict:
				token_dict[token] += 1
			else:
				token_dict[token] = 1
#		print(' '.join(tokens))
		if idx%10000 == 0:
			print("!!!!!!!!!!!!!!!!!!!!! --  %s %s"%(idx, len(token_dict)))

sorted_tokens_dict = sorted(token_dict.items(), key=operator.itemgetter(1))
with open('test.txt','w') as f:
	for key,value in sorted_tokens_dict:
		msg = '%s\t%s\n'%(key,value)
		f.write(msg)
		print(msg)

