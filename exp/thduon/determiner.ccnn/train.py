from framework.utils.data.text_indexer import TextIndexer
from word_classifier.data import ClassifierData
import framework.subgraph.losses as losses
import framework.utils.common as utils
from framework.trainer import Trainer
from time import time
import model
import os
import shutil

param_file = 'params.py'
params = utils.load_param_file(param_file)
if not "num_classes" in params:
	params['num_classes'] = len(params['keywords'])+1
indexer = TextIndexer.from_txt_file(utils.get_dict_value(params, 'vocab_file'),
																		max_size=utils.get_dict_value(params,'max_vocab_size',-1),
																		min_freq=utils.get_dict_value(params,'min_vocab_freq',-1))
indexer.add_token('<pad>')
if utils.get_dict_value(params, 'all_lowercase', False):
	indexer.add_token('<s>')
else:
	indexer.add_token('<s>')
indexer.add_token('unk')
os.makedirs(utils.get_dict_value(params,'output_location'), exist_ok=True)
indexer.save_vocab_as_pkl(os.path.join(utils.get_dict_value(params,'output_location'), 'vocab.pkl'))
shutil.copyfile(param_file,os.path.join(utils.get_dict_value(params,'output_location'), param_file))

params['vocab_size'] = indexer.vocab_size()
print("VOCAB SIZE: %s" % params['vocab_size'])
if 'training_data_dir' in params:
	training_data = ClassifierData.get_training_data(base_dir=params['training_data_dir'], indexer=indexer, params=params)
else:
	training_data = ClassifierData.get_monolingual_training(base_dir=params['monolingual_dir'],
																													indexer=indexer,
																													params=params)
def on_checkpoint_saved(trainer, params, save_path):
    msg = 'saved checkpoint: ' + save_path
    print(msg)

def train_iteration_done(trainer, epoch, index, iteration_count, loss_value, training_done, run_results, params):
	if iteration_count == 1:
		trainer._out_file = open(os.path.join(utils.get_dict_value(params,'output_location'), 'training_log.txt'), 'w')

	msg = ("%s, %s"%(time(), loss_value))
	print('%s: %s' % (iteration_count, msg))
	trainer._out_file.write('%s\n'%msg)
	trainer._out_file.flush()


#print(training_data.next_batch(10))
trainer = Trainer(inference=model.inference, batch_size=utils.get_dict_value(params, 'batch_size', 128),
                  loss=losses.softmax_xentropy
									, model_output_location=utils.get_dict_value(params, 'output_location')
									, name=utils.get_dict_value(params, 'model_name')
									, training_data=training_data,
                  params=params)

trainer.run(restore_latest_ckpt=False, save_network=True,
            save_ckpt=True, mini_batches_between_checkpoint=utils.get_dict_value(params, 'mini_batches_between_checkpoint', 1000),
            additional_nodes_to_evaluate=['encoded_sentence']
            ,on_checkpoint_saved=on_checkpoint_saved)

