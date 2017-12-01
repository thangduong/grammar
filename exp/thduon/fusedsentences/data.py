import random
def _gen_data(s1, s2, keywords=[','], num_before=5, num_after=5,
											 pad_tok="<pad>", null_sample_factor=0,
											 use_negative_only_data=True,
											 add_redundant_keyword_data=True,
											 start_token=None,
											 ignore_negative_data=False):
	yes_list = []
	null_list = []
	if start_token is not None and len(start_token) > 0:
		tokens = [pad_tok] * (num_before - 1) + [start_token]
	else:
		tokens = [pad_tok] * (num_before)
	tokens += s1 + s2 + [pad_tok] * num_after
	mark = num_before + len(s1)
	for toki in range(num_before, len(tokens) - num_after):
		s = tokens[(toki - num_before):(toki + num_after)]
		if mark == toki:
			l = 1
			yes_list.append([s, l])
		else:
			l = 0
			null_list.append([s, l])


def _gen_data_from_file(filename, keywords=[','], num_before=5, num_after=5,
											 pad_tok="<pad>", null_sample_factor=0,
											 use_negative_only_data=True,
											 add_redundant_keyword_data=True,
											 start_token=None,
											 ignore_negative_data=False,
											 gen_data_fcn = None):
	last_line = []
	with open(filename) as f:
		for line in f:
			yes_list = []
			null_list = []
			line = line.rstrip().lstrip().lower()
			line = line.split()
			if line[-1] == '.':
				line = line[:-1]
			if len(last_line)>0 and len(line) > 0:
				if start_token is not None and len(start_token)>0:
					tokens = [pad_tok]*(num_before-1) + [start_token]
				else:
					tokens = [pad_tok]*(num_before)
				tokens += line + last_line + [pad_tok]*num_after
				mark = num_before + len(line)
				for toki in range(num_before, len(tokens)-num_after):
					s = tokens[(toki-num_before):(toki+num_after)]
					if mark == toki:
						l = 1
						yes_list.append([s, l])
					else:
						l = 0
						null_list.append([s, l])

				if start_token is not None and len(start_token)>0:
					tokens = [pad_tok]*(num_before-1) + [start_token]
				else:
					tokens = [pad_tok]*(num_before)
				tokens += line + last_line + ['.'] + [pad_tok]*(num_after-1)
				mark = num_before + len(line)
				for toki in range(num_before, len(tokens)-num_after):
					s = tokens[(toki-num_before):(toki+num_after)]
					if mark == toki:
						l = 1
						yes_list.append([s, l])
					else:
						l = 0
						null_list.append([s, l])

				# reverse order
				if start_token is not None and len(start_token)>0:
					tokens = [pad_tok]*(num_before-1) + [start_token]
				else:
					tokens = [pad_tok]*(num_before)
				tokens += last_line + line + [pad_tok]*num_after
				mark = num_before + len(line)
				for toki in range(num_before, len(tokens)-num_after):
					s = tokens[(toki-num_before):(toki+num_after)]
					if mark == toki:
						l = 1
						yes_list.append([s, l])
					else:
						l = 0
						null_list.append([s, l])

				if start_token is not None and len(start_token)>0:
					tokens = [pad_tok]*(num_before-1) + [start_token]
				else:
					tokens = [pad_tok]*(num_before)
				tokens += last_line + line + ['.'] + [pad_tok]*(num_after-1)
				mark = num_before + len(line)
				for toki in range(num_before, len(tokens)-num_after):
					s = tokens[(toki-num_before):(toki+num_after)]
					if mark == toki:
						l = 1
						yes_list.append([s, l])
					else:
						l = 0
						null_list.append([s, l])
				random.shuffle(null_list)
				yes_list += null_list[:len(yes_list)]
				random.shuffle(yes_list)
				for r in yes_list:
					yield r
			last_line = line