import random
import math

def gen_data(tokens, keywords,
						 num_before=5, num_after=5,
						 pad_tok="<pad>", null_sample_factor=0,
						 add_redundant_keyword_data=True,
						 use_negative_only_data=True,
						 ignore_negative_data=False,
						 add_keyword_removal_data=True):
	# assume incoming tokens constitute a correct sentences
	tokens = [pad_tok] * num_before + tokens + [pad_tok]*num_after
	n0 = [] # set of sentences that need [keyword]
	n1 = [] # set of sentences that need a keyword
	n2 = [] # set of sentences that need [keyword] because there is one there already
	n3 = [] # there is a [keyword] and it should be removed
	if null_sample_factor < 0:
		null_sample_factor = 1/len(keywords)
	#print(tokens)
	class_offset = 1
	if ignore_negative_data:
		class_offset = 0
	for toki, tok in enumerate(tokens):
		if tok.lower() in keywords:
			idx = keywords.index(tok.lower())
			before_idx = toki - num_before
			after_idx = toki + num_after + 1
			#yield [tokens[before_idx:toki] + tokens[toki+1:after_idx], idx + 1]
			n1.append([tokens[before_idx:toki] + tokens[toki+1:after_idx], idx + class_offset])#tokens[before_idx:toki] + tokens[toki+1:after_idx])
			n2.append(tokens[(before_idx + 1):after_idx])
			n2.append(tokens[(before_idx):(after_idx-1)])
		elif toki > num_before and toki < len(tokens)-num_after and not ignore_negative_data:
			before_idx = toki - num_before
			after_idx = toki + num_after
			n0.append(tokens[before_idx:toki] + tokens[toki:after_idx])
			#yield [tokens[before_idx:toki] + tokens[toki+1:after_idx], 0]
			for keyword in keywords:
				n3.append(tokens[before_idx:toki] + [keyword] + tokens[toki:after_idx-1])
	if len(n1) > 0:
		if ignore_negative_data:
			n0 = []
		else:
			if null_sample_factor > 0:
				n0 = random.sample(n0, math.ceil(len(n1)*null_sample_factor))
	#		n0 = random.sample(n0, math.ceil(len(n1)))
		n = []
		n += [[x,0] for x in n0]
		n += [[x,y] for x,y in n1]
		if add_redundant_keyword_data:
			n += [[x,0] for x in n2]
		random.shuffle(n)
		for x in n:
			#print(x)
			yield x
	elif use_negative_only_data and not ignore_negative_data:
		n = []
		n += [[x,0] for x in n0]
		if add_redundant_keyword_data:
			n += [[x,0] for x in n2]
		random.shuffle(n)
		for x in n:
			#print(x)
			yield x

if __name__ == "__main__":
	sentence = 'this is the best night of my life'
	res = gen_data(sentence.split(), ['a','an','the','^a','^an','^the'])
	for s in res:
		print(s)