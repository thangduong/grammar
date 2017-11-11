from framework.utils.data.text_indexer import TextIndexer
from tellme.data import TellmeData
import framework.subgraph.losses as losses
import framework.utils.common as utils
from framework.trainer import Trainer
from time import time
import model
import os
import shutil

param_file = 'params.py'
params = utils.load_param_file(param_file)
os.makedirs(utils.get_dict_value(params,'output_location'), exist_ok=True)
shutil.copyfile(param_file,os.path.join(utils.get_dict_value(params,'output_location'), param_file))


training_data = TellmeData()
params['vocab_size'] = training_data.get_tcid_count()
params['num_classes'] = training_data.get_tcid_count()

def on_checkpoint_saved(trainer, params, save_path):
    msg = 'saved checkpoint: ' + save_path
    print(msg)

def train_iteration_done(trainer, epoch, index, iteration_count, loss_value, training_done, run_results, params):
	if iteration_count == 1:
		trainer._out_file = open(os.path.join(utils.get_dict_value(params,'output_location'), 'training_log.txt'), 'w')

	msg = ("%s, %s"%(time(), loss_value))
	print('%02d %04d: %s' % (epoch, iteration_count, msg))
	trainer._out_file.write('%s\n'%msg)
	trainer._out_file.flush()


#print(training_data.next_batch(10))
trainer = Trainer(inference=model.inference, batch_size=utils.get_dict_value(params, 'batch_size', 128),
                  loss=losses.softmax_xentropy
									, model_output_location=utils.get_dict_value(params, 'output_location')
									, name=utils.get_dict_value(params, 'model_name')
									, training_data=training_data, train_iteration_done=train_iteration_done,
                  params=params)

trainer.run(restore_latest_ckpt=False, save_network=True,
            save_ckpt=True, mini_batches_between_checkpoint=utils.get_dict_value(params, 'mini_batches_between_checkpoint', 1000),
            additional_nodes_to_evaluate=['encoded_sentence']
            ,on_checkpoint_saved=on_checkpoint_saved)
