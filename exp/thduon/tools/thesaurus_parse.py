import xml.etree.ElementTree
import operator

filename = "EnglishThesaurusData.xml"
e = xml.etree.ElementTree.parse(filename).getroot()
words = {}
for te in e:
	if te.tag=="thesaurus_entry":
		for tee in te:
			if tee.tag =="lexical_entry":
				word = tee.text
				if word not in words:
					words[word] = 1
			elif tee.tag=="sense":
				for s in tee:
#					print(s[0].text)
					word = s[0].text
					if word not in words:
						words[word] = 1


sorted_vocab = sorted(words.items(), key=operator.itemgetter(1), reverse=True)
maxl = 0
maxw = []
for x,y in sorted_vocab:
	l = len(x.split())
	if l>maxl:
		maxl = l
		maxw.append(x)
	print("%s %s" %(x,y))
print("maxl = %s"%maxl)
print(maxw)
