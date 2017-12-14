import sys
input_filename = sys.argv[1]
min_freq = 100
hard_keep_tokens =[
	',','.','"',"'",'(',')',
	'/','~','*','-','|','”','`','…','™','’','—',':','“','\\','=','°','+',',','"', '\'', '.','€'
]
with open(input_filename, 'r') as f:
	for line in f:
		line = line.rstrip().lstrip()
		pieces = line.split()
		if pieces[0] in hard_keep_tokens \
				or (pieces[0].isalpha() and 'ʼ' not in pieces[0] and int(pieces[1])>min_freq):
			print(line)
