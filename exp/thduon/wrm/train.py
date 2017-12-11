from framework.utils.data.text_indexer import TextIndexer
from word_classifier.data import ClassifierData
import framework.subgraph.losses as losses
import framework.utils.common as utils
import data
from framework.trainer import Trainer, _default_train_iteration_done
from time import time
import pickle
import model
import os
import shutil
import copy
import numpy as np

param_file = 'params.py'
params = utils.load_param_file(param_file)
params['num_classes'] = len(params['keywords'])+1
indexer = TextIndexer.from_txt_file(utils.get_dict_value(params, 'vocab_file'), max_size=utils.get_dict_value(params,'max_vocab_size',-1))
indexer.add_token('<pad>')
indexer.add_token('unk')
output_indexer = copy.deepcopy(indexer)
output_indexer.add_token('<blank>')
os.makedirs(utils.get_dict_value(params,'output_location'), exist_ok=True)
indexer.save_vocab_as_pkl(os.path.join(utils.get_dict_value(params,'output_location'), 'vocab.pkl'))

files_to_copy = [param_file]
for file in files_to_copy:
	shutil.copyfile(file,os.path.join(utils.get_dict_value(params,'output_location'), file))

params['vocab_size'] = indexer.vocab_size()

if 'training_data_dir' in params:
	training_data = ClassifierData.get_training_data(base_dir=params['training_data_dir'], indexer=indexer, params=params,
																													gen_data_fcn = data.gen_data)
else:
	training_data = ClassifierData.get_monolingual_training(base_dir=params['monolingual_dir'],
																													indexer=indexer,
																													params=params,
																													gen_data_fcn = data.gen_data)
live_replacement_count_filename = os.path.join(utils.get_dict_value(params,'output_location'), 'live_replacement_count.txt')
saved_replacement_count_filename = os.path.join(utils.get_dict_value(params,'output_location'), 'saved_replacement_count.txt')

def on_checkpoint_saved(trainer, params, save_path):
	msg = 'saved checkpoint: ' + save_path
	print(msg)
	save_y_count(trainer, saved_replacement_count_filename)

def save_y_count(trainer, filename = 'replacement_counts.txt'):
	with open(filename, 'w') as f:
		f.write("mean: %s\n" % trainer._training_data._mean)
		f.write("std: %s\n" % trainer._training_data._std)
		f.write("max: %s\n" % trainer._training_data._max)
		f.write("min: %s\n" % trainer._training_data._min)
		f.write("cap: %s\n" % (trainer._training_data._mean+2*trainer._training_data._std+100))
		total = np.sum(trainer._training_data._y_count[1:])
		for i, j in enumerate(trainer._training_data._y_count):
			if i > 0:
				f.write("%02d %05d %0.5f %s\n" % (i, j, j/total, params['id_to_keyword'][i - 1]))
			else:
				f.write("%02d %05d %0.5f %s\n" % (i, j, j/total, ''))


def train_iteration_done(trainer, epoch, index, iteration_count, loss_value, training_done, run_results, params):
	save_y_count(trainer, live_replacement_count_filename)
	_default_train_iteration_done(trainer, epoch, index, iteration_count, loss_value, training_done, run_results, params)


#print(training_data.next_batch(10))
trainer = Trainer(inference=model.inference, batch_size=utils.get_dict_value(params, 'batch_size', 128),
                  loss=losses.softmax_xentropy
									, model_output_location=utils.get_dict_value(params, 'output_location')
									, name=utils.get_dict_value(params, 'model_name')
									, training_data=training_data, train_iteration_done=train_iteration_done,
                  params=params)

trainer.run(restore_latest_ckpt=True, save_network=True,
            save_ckpt=True, mini_batches_between_checkpoint=utils.get_dict_value(params, 'mini_batches_between_checkpoint', 1000),
            additional_nodes_to_evaluate=['encoded_sentence']
            ,on_checkpoint_saved=on_checkpoint_saved)

