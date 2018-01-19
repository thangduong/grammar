# copy model to repo after release is called
from shell_command import shell_call
import framework.utils.common as utils
import os
import sys
import gflags


def callcmd(cmd):
	print("executing: %s"%cmd)
	shell_call(cmd)

def copy2repo(source_dir, target_dir, model_name):
	"""
	Copy from <source_dir>/<model_name> to <target_dir>/<model_name> if doesn't exist in target.
	If exists in target, then copy <source_dir>/<model_name>/release to <target_dir>/<model_name>/release.<num>
	where <num> is the next unique number.
	@param source_dir: source directory
	@param target_dir: target directory
	@param model_name: name of model
	@return: <num>, -1 if source_model_dir doesn't exist, -2 if no release dir in source_model_dir
	"""
	target_model_dir = os.path.join(target_dir, model_name)
	source_model_dir = os.path.join(source_dir, model_name)
	if not os.path.isdir(source_model_dir):
		return -1, source_model_dir
	if not os.path.isdir(target_dir):
		return -3, target_dir

	cmd = 'cp -rvf \"%s\" \"%s\"'%(source_model_dir,target_dir)
	callcmd(cmd)

	release_num = 0
	found = False
	while not found:
		target_release_dir = os.path.join(target_model_dir, 'release.%s'%release_num)
		if os.path.isdir(target_release_dir):
			release_num+=1
		else:
			found=True

	source_release_dir = os.path.join(source_model_dir, 'release')
	if not os.path.isdir(source_release_dir):
		return -2, source_release_dir

	cmd = 'cp -rvf \"%s\" \"%s\"'%(source_release_dir,target_release_dir)
	callcmd(cmd)
	return release_num, None


def main(argv):
	try:
		argv = FLAGS(argv)  # parse flags
	except gflags.FlagsError as e:
		print('%s\\nUsage: %s ARGS\\n%s' % (e, sys.argv[0], FLAGS))
		sys.exit(1)

	params = utils.load_param_file(FLAGS.paramsfile)
	model_dirname = os.path.dirname(FLAGS.paramsfile)
	src_dir = os.path.abspath(os.path.join(model_dirname, '..'))
	target_dir = FLAGS.target_dir
	model_name = params['model_name']
	result, dir_name = copy2repo(src_dir, target_dir, model_name)
	if result == -1:
		print("source dir %s doesn't exist" % dir_name)
	elif result == -2:
		print("release source dir %s doesn't exist" % dir_name)
	elif result == -3:
		print("target dir %s doesn't exist" % dir_name)
	else:
		print("SUCCESSFULLY COPY MODEL %s.%s TO REPO"%(model_name, result))

if __name__ == '__main__':
	gflags.DEFINE_string('paramsfile', 'params.py', 'parameters file')
	gflags.DEFINE_string('target_dir', '/models/dlframework_models', 'directory of repo')
	FLAGS = gflags.FLAGS
	main(sys.argv)
