import framework.utils.common as utils
import matplotlib.pyplot as plt
import numpy as np
import csv
import gflags
import os
import sys

gflags.DEFINE_string('paramsfile', 'output/tellmeV5/params.py', 'parameter files')
FLAGS = gflags.FLAGS

def load_results(params):
	traininglog_file = os.path.join(utils.get_dict_value(params,'output_location'),
											'training_log.txt')
	data = []
	with open(traininglog_file, 'r') as f:
		csvdata = csv.reader(f, delimiter=',')
		for row in csvdata:
			for i, col in enumerate(row):
				if i >= len(data):
					data.append([])
				data[i].append(float(col))
	return data


def main(argv):
	try:
		argv = FLAGS(argv)  # parse flags
	except gflags.FlagsError as e:
		print('%s\\nUsage: %s ARGS\\n%s' % (e, sys.argv[0], FLAGS))
		sys.exit(1)
	print(FLAGS.paramsfile)
	params = utils.load_param_file(FLAGS.paramsfile)
	data = load_results(params)
	fig, ax = plt.subplots()
	ax.plot([(x * 8192)/1000000 for x in data[1]], data[8])

	ax.set(xlabel='Million Records Seen', ylabel='Accuracy @ 1',
				 title=params['model_name'])
	ax.grid()

	fig.savefig("test.png")
	plt.ylim((.75,.78))
	#plt.ylim((.75,1))
	plt.show()
#	print(data[1])


if __name__ == '__main__':
	main(sys.argv)