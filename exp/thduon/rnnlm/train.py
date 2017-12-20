from framework.utils.data.text_indexer import TextIndexer
from data import RnnLmData
import framework.subgraph.losses as losses
import framework.utils.common as utils
from framework.trainer import Trainer
from time import time
import model
import os
import shutil

param_file = 'params.py'
params = utils.load_param_file(param_file)
indexer = TextIndexer.from_txt_file(utils.get_dict_value(params, 'vocab_file'), max_size=utils.get_dict_value(params,'max_vocab_size',-1))
if utils.get_dict_value(params, 'all_lowercase', False):
	indexer.add_token('<s>')
else:
	indexer.add_token('<S>')
indexer.add_token('unk')
os.makedirs(utils.get_dict_value(params,'output_location'), exist_ok=True)
indexer.save_vocab_as_pkl(os.path.join(utils.get_dict_value(params,'output_location'), 'vocab.pkl'))
shutil.copyfile(param_file,os.path.join(utils.get_dict_value(params,'output_location'), param_file))
params['vocab_size'] = indexer.vocab_size()

if 'training_data_dir' in params:
	data = RnnLmData(indexer=indexer, params=params)
else:
	print("No data dir in params file")
	exit(0)

def on_checkpoint_saved(trainer, params, save_path):
	msg = 'saved checkpoint: ' + save_path
	print(msg)

#print(training_data.next_batch(10))
trainer = Trainer(inference=model.inference, batch_size=utils.get_dict_value(params, 'batch_size', 128),
                  loss=model.loss
									, model_output_location=utils.get_dict_value(params, 'output_location')
									, name=utils.get_dict_value(params, 'model_name')
									, training_data=data,
                  params=params)

trainer.run(restore_latest_ckpt=True, save_network=True,
            save_ckpt=True, mini_batches_between_checkpoint=utils.get_dict_value(params, 'mini_batches_between_checkpoint', 1000)
            ,on_checkpoint_saved=on_checkpoint_saved)

