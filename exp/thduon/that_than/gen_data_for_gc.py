
def gen_candidates(tokens, num_before, num_after, pad_tok):
	num_tokens = len(tokens)
	tokens = [pad_tok] * num_before + tokens + [pad_tok]*num_after
	n0 = []
	n1 = []
	for toki in range(0, num_tokens):
			n0.append(tokens[toki:(toki + num_before + num_after)])
	return n0

x = ['technical', 'failure', 'in', 'the', 'lift', 'system', 'at', 'the', 'alpine', 'Brauneck', 'resort', 'in', 'Bavaria', 'triggered', 'the', 'automatic', 'shutdown', 'of', 'the', 'whole']
gen_candidates(x,5,5,'<pad>')