TOKEN_DELIMITERS = ['/','~','*','-','|','”','`','…','™','’','—',':','“','\\','=','°','+',',','"', '\'', '.','€',
									'(', ')',',','&','!','#', '$','?',';','ʼ', "'"
									, b'\xe2\x80\x80'.decode('utf-8')
									, b'\xe2\x80\x81'.decode('utf-8')
									, b'\xe2\x80\x82'.decode('utf-8')
									, b'\xe2\x80\x83'.decode('utf-8')
									, b'\xe2\x80\x84'.decode('utf-8')
									, b'\xe2\x80\x85'.decode('utf-8')
									, b'\xe2\x80\x86'.decode('utf-8')
									, b'\xe2\x80\x87'.decode('utf-8')
									, b'\xe2\x80\x88'.decode('utf-8')
									, b'\xe2\x80\x89'.decode('utf-8')
									, b'\xe2\x80\x8a'.decode('utf-8')
									, b'\xe2\x80\x8b'.decode('utf-8')
									, b'\xe2\x80\x8c'.decode('utf-8')
									, b'\xe2\x80\x8d'.decode('utf-8')
									, b'\xe2\x80\x8e'.decode('utf-8')
									, b'\xe2\x80\x8f'.decode('utf-8')
									, b'\xe2\x80\x90'.decode('utf-8')
									, b'\xe2\x80\x91'.decode('utf-8')
									, b'\xe2\x80\x92'.decode('utf-8')
									, b'\xe2\x80\x93'.decode('utf-8')
									, b'\xe2\x80\x94'.decode('utf-8')
									, b'\xe2\x80\x95'.decode('utf-8')
									, b'\xe2\x80\x96'.decode('utf-8')
									, b'\xe2\x80\x97'.decode('utf-8')
									, b'\xe2\x80\x98'.decode('utf-8')
									, b'\xe2\x80\x99'.decode('utf-8')
									, b'\xe2\x80\x9a'.decode('utf-8')
									, b'\xe2\x80\x9b'.decode('utf-8')
									, b'\xe2\x80\x9c'.decode('utf-8')
									, b'\xe2\x80\x9d'.decode('utf-8')
									, b'\xe2\x80\x9e'.decode('utf-8')
									, b'\xe2\x80\x9f'.decode('utf-8')
									, b'\xe2\x80\xa0'.decode('utf-8')
									, b'\xe2\x80\xa1'.decode('utf-8')
									, b'\xe2\x80\xa2'.decode('utf-8')
									, b'\xe2\x80\xa3'.decode('utf-8')
									, b'\xe2\x80\xa4'.decode('utf-8')
									, b'\xe2\x80\xa5'.decode('utf-8')
									, b'\xe2\x80\xa6'.decode('utf-8')
									, b'\xe2\x80\xa7'.decode('utf-8')
									, b'\xe2\x80\xa8'.decode('utf-8')
									, b'\xe2\x80\xa9'.decode('utf-8')
									, b'\xe2\x80\xaa'.decode('utf-8')
									, b'\xe2\x80\xab'.decode('utf-8')
									, b'\xe2\x80\xac'.decode('utf-8')
									, b'\xe2\x80\xad'.decode('utf-8')
									, b'\xe2\x80\xae'.decode('utf-8')
									, b'\xe2\x80\xaf'.decode('utf-8')
									, b'\xe2\x80\xb0'.decode('utf-8')
									, b'\xe2\x80\xb1'.decode('utf-8')
									, b'\xe2\x80\xb2'.decode('utf-8')
									, b'\xe2\x80\xb3'.decode('utf-8')
									, b'\xe2\x80\xb4'.decode('utf-8')
									, b'\xe2\x80\xb5'.decode('utf-8')
									, b'\xe2\x80\xb6'.decode('utf-8')
									, b'\xe2\x80\xb7'.decode('utf-8')
									, b'\xe2\x80\xb8'.decode('utf-8')
									, b'\xe2\x80\xb9'.decode('utf-8')
									, b'\xe2\x80\xba'.decode('utf-8')
									, b'\xe2\x80\xbb'.decode('utf-8')
									, b'\xe2\x80\xbc'.decode('utf-8')
									, b'\xe2\x80\xbd'.decode('utf-8')
									, b'\xe2\x80\xbe'.decode('utf-8')
									, b'\xe2\x80\xbf'.decode('utf-8')
]

def reconstitute_token(tok):
	if type(tok) is list or type(tok) is tuple:
		result = ''
		for subtok in tok:
			if subtok == "'s" or subtok == "n't" or len(result)==0:
				result += subtok
			else:
				result += ' ' + subtok
		return result
	else:
		return tok


def tokenize(txt,
						 skip_separators=[' ','\n','\r'],
						 keep_separators=TOKEN_DELIMITERS,
						 verbose=True):
	if verbose:
		print("TOKENIZING: [%s]" % txt)
	tokens = []
	pos = []
	i = 0
	state = 0
	start = 0
	while (i < len(txt)):
		# TODO: deal with this in a smart and general way
		if txt[i:(i+3)] == "n't":
			if state == 1:
				tokens.append(txt[start:i])
				pos.append(start)
			tokens.append(txt[i:(i+3)])
			pos.append(i)
			i += 3
			state = 0
		elif txt[i:(i + 2)] == "'s":
			if state == 1:
				tokens.append(txt[start:i])
				pos.append(start)
			tokens.append(txt[i:(i + 2)])
			pos.append(i)
			i += 2
			state = 0
		elif txt[i] in skip_separators:
			if state == 1:
				tokens.append(txt[start:i])
				pos.append(start)
			while (i < len(txt)) and txt[i] in skip_separators:
				i += 1
			state = 0
		elif txt[i] in keep_separators:
			if state == 1:
				tokens.append(txt[start:i])
				pos.append(start)
			tokens.append(txt[i])
			pos.append(i)
			i += 1
			state = 0
		else:
			if state == 0:
				start = i
				state = 1
			i += 1
	if state == 1:
		tokens.append(txt[start:i])
		pos.append(start)
	if verbose:
		print(tokens)
	return tokens, pos
