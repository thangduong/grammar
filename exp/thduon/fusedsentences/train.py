from framework.utils.data.text_indexer import TextIndexer
from word_classifier.data import ClassifierData
import framework.subgraph.losses as losses
import framework.utils.common as utils
from framework.trainer import Trainer, _default_train_iteration_done
from data import _gen_data_from_file, _gen_data
from time import time
import model
import os
import shutil

param_file = 'params.py'
params = utils.load_param_file(param_file)
params['num_classes'] = len(params['keywords'])+1
indexer = TextIndexer.from_txt_file(utils.get_dict_value(params, 'vocab_file'))
indexer.add_token('<pad>')
indexer.add_token('unk')
print("VOCAB SIZE=%s"%indexer.vocab_size())
os.makedirs(utils.get_dict_value(params,'output_location'), exist_ok=True)
indexer.save_vocab_as_pkl(os.path.join(utils.get_dict_value(params,'output_location'), 'vocab.pkl'))
shutil.copyfile(param_file,os.path.join(utils.get_dict_value(params,'output_location'), param_file))

params['vocab_size'] = indexer.vocab_size()
training_data = ClassifierData.get_monolingual_training(base_dir=params['monolingual_dir'],
																												indexer=indexer,
																												params=params,
																												gen_data_from_file_fcn=_gen_data_from_file,
																												gen_data_fcn=_gen_data)
def on_checkpoint_saved(trainer, params, save_path):
    msg = 'saved checkpoint: ' + save_path
    print(msg)


#print(training_data.next_batch(10))
trainer = Trainer(inference=model.inference, batch_size=utils.get_dict_value(params, 'batch_size', 128),
                  loss=losses.softmax_xentropy
									, model_output_location=utils.get_dict_value(params, 'output_location')
									, name=utils.get_dict_value(params, 'model_name')
									, training_data=training_data, train_iteration_done=_default_train_iteration_done,
                  params=params)

trainer.run(restore_latest_ckpt=False, save_network=True,
            save_ckpt=True, mini_batches_between_checkpoint=utils.get_dict_value(params, 'mini_batches_between_checkpoint', 1000),
            additional_nodes_to_evaluate=['encoded_sentence']
            ,on_checkpoint_saved=on_checkpoint_saved)

