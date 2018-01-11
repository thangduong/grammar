import time
import os
import sys

cmd = 'python3 eval_heldout.py %s' % sys.argv[1]
while True:
	os.system(cmd)
	time.sleep(60*10)
