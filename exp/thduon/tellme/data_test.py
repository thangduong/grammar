# test code for the data reader
from data import TellmeData
td = TellmeData()
b = td.next_batch(batch_size=1000000)
bs = (len(b['y']))
e = td.current_epoch()
laste = e
while e == laste:
	laste = e
	print("%s-%s"%(e,bs))
	b = td.next_batch(batch_size=1000000)
	bs = (len(b['y']))
	e = td.current_epoch()
print("%s-%s"%(e,bs))