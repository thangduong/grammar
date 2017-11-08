from framework.utils.data.text_indexer import TextIndexer
from classifier_data import ClassifierData
import framework.subgraph.losses as losses
import framework.utils.common as utils
from framework.trainer import Trainer
from time import time
import model
import os

MODEL_NAME = "v0"

params = { 'num_words_before': 5,
					 'num_words_after': 5,
					 'embedding_size': 300,
					 'vocab_size': 100000,
					 'embedding_device': None,
					 'batch_size': 128,
					 'num_classes': 2,
					 'mini_batches_between_checkpoint': 100,
					 'monolingual_dir': '/mnt/work/1-billion-word-language-modeling-benchmark'
					 }

indexer = TextIndexer.from_txt_file(os.path.join(params['monolingual_dir'], '1b_word_vocab.txt'))
indexer.add_token('<pad>')
indexer.add_token('unk')
indexer.save_vocab_as_pkl('vocab.pkl')
params['vocab_size'] = indexer.vocab_size()
training_data = ClassifierData.get_monolingual_training(base_dir=params['monolingual_dir'],
																												indexer=indexer,
																												params=params)
#print(training_data.next_batch())
def on_checkpoint_saved(trainer, params, save_path):
    msg = 'saved checkpoint: ' + save_path
    print(msg)
#    params['logfile'].write(msg)
#    params['logfile'].write('\n')

def train_iteration_done(trainer, epoch, index, iteration_count, loss_value, training_done, run_results, params):
	if iteration_count == 1:
		trainer._out_file = open('output.txt', 'w')

	msg = ("%s, %s"%(time(), loss_value))
	print('%s: %s' % (iteration_count, msg))
	trainer._out_file.write('%s\n'%msg)

trainer = Trainer(inference=model.inference, batch_size=utils.get_dict_value(params, 'batch_size', 128),
                  loss=losses.softmax_xentropy, model_output_location="./output",
                  name=MODEL_NAME, training_data=training_data, train_iteration_done=train_iteration_done,
                  params=params)

trainer.run(restore_latest_ckpt=False, save_network=True,
            save_ckpt=True, mini_batches_between_checkpoint=utils.get_dict_value(params, 'mini_batches_between_checkpoint', 1000),
            additional_nodes_to_evaluate=['encoded_sentence']
            ,on_checkpoint_saved=on_checkpoint_saved)

