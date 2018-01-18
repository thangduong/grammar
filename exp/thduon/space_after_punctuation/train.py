from data import Data
import framework.subgraph.losses as losses
import framework.utils.common as utils
from framework.trainer import Trainer, _default_train_iteration_done
from time import time
import model
import os
import shutil

param_file = 'params.py'
params = utils.load_param_file(param_file)
os.makedirs(utils.get_dict_value(params,'output_location'), exist_ok=True)
shutil.copyfile(param_file,os.path.join(utils.get_dict_value(params,'output_location'), param_file))

if 'training_data_dir' in params:
	data = Data.get_data(params, params['training_data_dir'])

def on_checkpoint_saved(trainer, params, save_path):
    msg = 'saved checkpoint: ' + save_path
    print(msg)


#print(training_data.next_batch(10))
trainer = Trainer(inference=model.inference, batch_size=utils.get_dict_value(params, 'batch_size', 128),
                  loss=losses.softmax_xentropy
									, model_output_location=utils.get_dict_value(params, 'output_location')
									, name=utils.get_dict_value(params, 'model_name')
									, training_data=data, train_iteration_done=_default_train_iteration_done,
                  params=params)

trainer.run(restore_latest_ckpt=False, save_network=True,
            save_ckpt=True, mini_batches_between_checkpoint=utils.get_dict_value(params, 'mini_batches_between_checkpoint', 1000),
            additional_nodes_to_evaluate=['encoded_sentence']
            ,on_checkpoint_saved=on_checkpoint_saved)
