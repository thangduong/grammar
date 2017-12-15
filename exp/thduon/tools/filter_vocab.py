import sys
import re
input_filename = sys.argv[1]
min_freq = 100
hard_keep_tokens =[
	',','.','"',"'",'(',')',
	'/','~','*','-','|','”','`','…','™','’','—',':','“','\\','=','°','+',',','"', '\'', '.','€',
	"'s", "n't", "'t",'$','mr.','mrs.','dr.', '--', ';','?','e.g.','i.e.'
]
with open(input_filename, 'r') as f:
	for line in f:
		line = line.rstrip().lstrip()
		pieces = line.split()
		if int(pieces[1])>min_freq and len(pieces[0])<25 and not(re.match('\d',pieces[0])) \
				and not(pieces[0].endswith('.com')):
			print(line)
			continue
#		if pieces[0] in hard_keep_tokens \
#				or (pieces[0].isalpha() and 'ʼ' not in pieces[0] and int(pieces[1])>min_freq):
#			print(line)
