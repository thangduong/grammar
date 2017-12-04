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
	random.shuffle(null_list)
	yes_list += null_list[:(5*(len(yes_list)+1))]
	random.shuffle(yes_list)
	for r in yes_list:
		yield r


def _gen_data_from_file(filename, keywords=[','], num_before=5, num_after=5,
											 pad_tok="<pad>", null_sample_factor=0,
											 use_negative_only_data=True,
											 add_redundant_keyword_data=True,
											 start_token=None,
											 ignore_negative_data=False,
											 gen_data_fcn = None):
	group_len = 500
	line_group = []
	with open(filename) as f:
		done = False
		while not done:
			line = next(f, None)
			if line is not None:
				line = line.rstrip().lstrip().lower()
				line = line.split()
				if line[-1] == '.':
					line = line[:-1]
				if len(line) >= num_before:
					continue
				line_group.append(line)
			else:
				done = True
			if (done and len(line_group)>0) or (len(line_group)==group_len):
				for i in range(len(line_group)):
					yield from _gen_data(line_group[i], [], keywords, num_before, num_after,
															 pad_tok, null_sample_factor,
															 use_negative_only_data,
															 add_redundant_keyword_data,
															 start_token,
															 ignore_negative_data)
					for j in range(len(line_group)):
						yield from _gen_data(line_group[i], line_group[j], keywords, num_before, num_after,
											 pad_tok, null_sample_factor,
											 use_negative_only_data,
											 add_redundant_keyword_data,
											 start_token,
											 ignore_negative_data)
				line_group = []