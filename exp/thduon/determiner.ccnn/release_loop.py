import time
import os
import sys

cmd = 'python3 ../tools/release_model.py %s' % sys.argv[1]
while True:
	os.system(cmd)
	time.sleep(60*60*2)
