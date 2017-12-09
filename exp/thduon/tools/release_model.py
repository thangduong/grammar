# build release package
from shell_command import shell_call
import sys
import os
paramsfile = sys.argv[1]
model_dirname = os.path.dirname(paramsfile)
script_path = os.path.dirname(sys.argv[0])
freeze_cmd = 'python3 %s/freeze_model.py %s' % (script_path, os.path.join(model_dirname,'params.py'))
vocab_cmd = 'python3 %s/pkl2json.py %s' % (script_path, os.path.join(model_dirname,'vocab.pkl'))
params_cmd = 'python3 %s/params2json.py %s' % (script_path, paramsfile)
release_cmd = 'mkdir %s' % os.path.join(model_dirname, "release")
release_cmd2 = 'cp -rvf %s %s %s' % (
									model_dirname + '/*.graphdef',
									model_dirname + '/*.json',
									os.path.join(model_dirname, "release")
								)

cmds = [freeze_cmd, vocab_cmd, params_cmd, release_cmd, release_cmd2]
for c in cmds:
	shell_call(c)