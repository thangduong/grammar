# plot logs and save as png files
import os
import csv
import gflags
import sys
import numpy as np
import matplotlib.pyplot as plt

gflags.DEFINE_string('model_name', 'determinerCCNNV57', 'name of model')
gflags.DEFINE_string('models_dir', '/models/dlframework_models', 'dir where models are stored')
FLAGS = gflags.FLAGS



def load_results(filename, delimiter=','):
	data = []
	with open(filename, 'r') as f:
		csvdata = csv.reader(f, delimiter=delimiter)
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

	model_path = os.path.join(FLAGS.models_dir, FLAGS.model_name)
	scores_file = os.path.join(model_path, 'scores.txt')
	training_log_file = os.path.join(model_path, 'training_log.txt')
	print("Loading scores...")
	scores = load_results(scores_file, delimiter=' ')
	print("Loading training log...")
	training_log = load_results(training_log_file)


	# Plot the loss
	fig, ax = plt.subplots()
	ax.plot([(x * 8192)/1000000 for x in training_log[1]], training_log[3])

	ax.set(xlabel='Million Records Seen', ylabel='Loss',
				 title=FLAGS.model_name)
	ax.grid()

	min_value = np.min(training_log[3])
#	plt.ylim((math.floor(min_value*10)/10,1))
	#plt.ylim((.75,1))
	fig.savefig(os.path.join(model_path,"loss.png"))
	plt.show(block=False)



	# Plot the accuracy
	fig, ax = plt.subplots()
	y = scores[-1]
	x = [(x - scores[0][0]) / (60*60) for x in scores[0]]
	ax.plot(x, y)

	ax.set(xlabel='Time Trained (hr)', ylabel='Acurracy (%)',
				 title=FLAGS.model_name)
	ax.grid()

	min_value = np.min(y)
	max_value = np.max(y)
	plt.ylim((np.max([max_value*.99,min_value]),max_value))
	fig.savefig(os.path.join(model_path, "accuracy.png"))
	plt.show(block=False)

	input("Press enter to exit...")
#	print(data[1])


if __name__ == '__main__':
	main(sys.argv)
