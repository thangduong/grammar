import framework.utils.common as utils
import matplotlib.pyplot as plt
import numpy as np
import math
import csv
import gflags
import os
import sys


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
	params_files_list = ['output/tellmeV12/params.py','output/tellmeV15/params.py'
	, 'output/tellmeV17/params.py'
	, 'output/tellmeV18/params.py'
		, 'output/tellmeV19/params.py']

	params = []
	data = []
	for paramsfile in params_files_list:
		params.append(utils.load_param_file(paramsfile))
		data.append(load_results(params[-1]))

	fig, ax = plt.subplots()

	for cdata in data:
		ax.plot([(x * 8192)/1000000 for x in cdata[1]], cdata[8])

	ax.set(xlabel='Million Records Seen', ylabel='Accuracy @ 1',
				 title=','.join([x['model_name'] for x in params]))
	ax.grid()

	max_value = .82#np.max(data[8])
	#plt.ylim((.75,math.ceil(max_value*10)/10))
	plt.ylim((.75,.81))
	#plt.ylim((.75,1))
	fig.savefig("accuracy_compare.png")
	plt.show(block=False)

	fig, ax = plt.subplots()
	for cdata in data:
		ax.plot([(x * 8192)/1000000 for x in cdata[1]], cdata[3])

	ax.set(xlabel='Million Records Seen', ylabel='Loss',
				 title=','.join([x['model_name'] for x in params]))
	ax.grid()

	min_value = 0#np.min(data[3])
	plt.ylim((math.floor(min_value*10)/10,1))
	#plt.ylim((.75,1))
	fig.savefig("loss_compare.png")
	plt.show(block=False)


	input("Press enter to exit...")
#	print(data[1])


if __name__ == '__main__':
	main(sys.argv)