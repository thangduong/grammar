import framework.subgraph.losses as losses
import framework.utils.common as utils
from framework.trainer import Trainer
import model
import data
import os
import shutil

param_file = 'params.py'
params = utils.load_param_file(param_file)
os.makedirs(utils.get_dict_value(params,'output_location'), exist_ok=True)

files_to_copy = [param_file]
for file in files_to_copy:
	shutil.copyfile(file,os.path.join(utils.get_dict_value(params,'output_location'), file))

files = []
for i in range(29):
	files.append(['features_%03d.npy'%i,'scores_%03d.npy'%i])
training_data = data.PPTDesignerData(params=params, files=files)

trainer = Trainer(inference=model.inference, batch_size=utils.get_dict_value(params, 'batch_size', 128),
                  loss=losses.l2_loss
									, model_output_location=utils.get_dict_value(params, 'output_location')
									, name=utils.get_dict_value(params, 'model_name')
									, training_data=training_data,
                  params=params)

trainer.run(restore_latest_ckpt=False, save_network=True,
            save_ckpt=True, mini_batches_between_checkpoint=utils.get_dict_value(params, 'mini_batches_between_checkpoint', 1000))

