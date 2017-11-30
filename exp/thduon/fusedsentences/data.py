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
			line = line.rstrip().lstrip()
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
					else:
						l = 0
					yield [s,l]
			last_line = line