import simplediff as sd
import json
import requests

def get_critique(sentence, conf):
	req = {'sentence': sentence}
	msg = json.dumps(req)
#	print(msg)
	r = requests.post("http://oxo-grammar-demo2.thangduong.com/critique", data=msg)
	r = json.loads(r.text)
	result = []
	for x in r['critiques']:
		if x['conf']> conf:
			result.append([x['conf'],[x['tstart'], x['src_word'].rstrip().lstrip(), x['tgt_word'].rstrip().lstrip()]])
	return result

#cr = get_critique("the quick brown fox jumped over the lazy dog")
#print(cr)
#exit(0)


f1n = 'wikiAny50KTest.enz.tok.txt'
f2n = 'wikiAny50KTest.enu.tok.txt'
cache_pkl = "wikiAny50K.pkl"
#f1n = 'wiki2017CleanChainLifetime.enz.tok.txt'
#f2n = 'wiki2017CleanChainLifetime.enu.tok.txt'
#cache_pkl = 'wiki_peipei.pkl'
#f1n = 'clc.enz.tok.txt'
#f2n = 'clc.enu.tok.txt'
#cach_pkl = 'clc_analyzed.pkl'
#f1n = 'wikiAny3000000Test.enz.tok.txt'
#f2n = 'wikiAny3000000Test.enu.tok.txt'
#cach_pkl = 'wikiAny3000000.pkl'
wordlist = ['','a','an','the']
critique_count = 0
cmatrix = [[0]*4 for i in range(4)]
repeat_count = 0
gt_data = []
with open(f1n) as f1, open(f2n) as f2:
	for l1, l2 in zip(f1,f2):
		l1 = l1.rstrip().lstrip()
		l2 = l2.rstrip().lstrip()
		deltas = sd.string_diff(l1,l2)
		has_correct_delta = False
#		for deltai, delta in enumerate(deltas):
		deltai = 0
		cur_critique_count = 0
		tstart = 0
		gt = {'sentence':l1, 'critiques':[]}
		while deltai < len(deltas):
			delta = deltas[deltai]
			if (delta[0] == '-' or delta[0] == '+') and (len(delta[1])==1) and (delta[1][0] in wordlist):
				source  = ''
				target = ''
				cur_critique_count += 1
				is_valid_critique = True
				if delta[0] == '-' and deltai < len(deltas)-1 and deltas[deltai+1][0] == '+':
					target = deltas[deltai+1][1][0]
					source = delta[1][0]
					if target not in wordlist:
						# transition to something else
						cur_critique_count -= 1
						is_valid_critique = False
					else:
#						cur_critique_count -= 1
#						is_valid_critique = False
						print('%03d %s -> %s' % (tstart, source, target))
					deltai += 1
				elif delta[0] == '+' and deltai > 0 and deltas[deltai - 1][0] == '-':
					# transition from something else
					target = delta[1][0]
					source = deltas[deltai-1][1][0]
					cur_critique_count -= 1
					is_valid_critique = False
#					print('**** %s -> %s' % (source, target))
				else:
					if delta[0] == '-':
						source = delta[1][0]
						if (deltai > 0 and deltas[deltai-1][1][-1]==source) or \
							(deltai < len(deltas)-1 and (deltas[deltai+1][1][0] == source)):
							print("REPEAT")
							repeat_count += 1
					else:
						target = delta[1][0]
					print('%03d %s %s' % (tstart, delta[0], delta[1][0]))
				if is_valid_critique:
					gt['critiques'].append([tstart, source, target])
				if is_valid_critique:
					cmatrix[wordlist.index(source)][wordlist.index(target)] += 1
			if delta[0] == '=' or delta[0] == '-':
				tstart += len(delta[1])
			deltai += 1
		if len(gt['critiques'])>0:
			gt_data.append(gt)
		critique_count += cur_critique_count
		if cur_critique_count > 0:
			print('>%s\n<%s' % (l1, l2))
			print(deltas)
			print("\n")

print("GENERATING CRITIQUES FROM SERVER!!!")
for gt in gt_data:
	g = gt['critiques']
	c = get_critique(gt['sentence'], conf=0)
	gt['model_critiques'] = c
	print(gt['sentence'])
	print(g)
	print(c)
	print("\n")

print("critique count = %s"%critique_count)
print("repeat count = %s"%repeat_count)
for r in cmatrix:
	rstr = ''
	for c in r:
		rstr += ' %04d'%c
	print(rstr)


import pickle
with open(cache_pkl,'wb') as f:
	pickle.dump(gt_data, f)

plist = []
rlist = []
for conf in [10,20,30,40,50,60,70,80,90,95,96,97,98,99,99.5]:
	num_right = 0
	num_returned = 0
	num_total = 0
	for gt in gt_data:
		g = gt['critiques']
		c = [x[1] for x in gt['model_critiques'] if x[0] > conf]
			#get_critique(gt['sentence'], conf)
		num_returned += len(c)
		num_total += len(g)

		rcount = 0
		for cc in c:
			if cc in g:
				rcount+=1
#		print(g)
#		print(c)
#		print(rcount)
		num_right += rcount

	print(num_right)
	print(num_returned)
	print(num_right/(num_returned-repeat_count))
	print(num_total)
	print(num_right/num_total)
	plist.append(num_right/(num_returned-repeat_count))
	rlist.append(num_right/num_total)

for p,r in zip(plist,rlist):
	print("%0.4f %0.4f"%(p,r))
