from tokenex.tokenizer import split_token
import operator
import pathlib
import pickle
import gflags
import os
import sys
import json

gflags.DEFINE_string('data_dir', '/mnt/work/training_data.tok2', 'directory containing data')
gflags.DEFINE_string('output_dir', '', 'directory to put output file.  if "", then use data_dir/vocab')
FLAGS = gflags.FLAGS

def gen_vocab(data_dir):
	filenames = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir,f))]
	vocab = {}
	lowercase_vocab = {}
	for filename in filenames:
		print("File: %s" % filename)
		with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
			for line in f:
				line = line.rstrip().lstrip()
				tokens = line.split()
				for token in tokens:
					token, _ = split_token(token)
					if token in vocab:
						vocab[token] += 1
					else:
						vocab[token] = 1
					token = token.lower()
					if token in lowercase_vocab:
						lowercase_vocab[token] += 1
					else:
						lowercase_vocab[token] = 1
	return vocab, lowercase_vocab

def filter_vocab(vocab, min_count):
	words_to_del = []
	for word in vocab.keys():
		if vocab[word] < min_count:
			words_to_del.append(word)
	for word in words_to_del:
		del vocab[word]

def save_vocab(file_path, vocab):
	sorted_vocab = sorted(vocab.items(), key=operator.itemgetter(1), reverse=True)
	filename = '%s.pkl'%file_path
	print("Writing %s" % filename)
	with open(filename,'wb') as f:
		pickle.dump(vocab, f)
	filename = '%s.txt'%file_path
	print("Writing %s" % filename)
	with open(filename, 'w') as f:
		for x,y in sorted_vocab:
			f.write('%s %s\n'%(x,y))

def save_vocabs(output_dir, vocab, lowercase_vocab, postfix):
	pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
	vocab_fn = os.path.join(output_dir, "vocab")
	lowercase_vocab_fn = os.path.join(output_dir, "lowercase_vocab")
	if len(postfix)>0:
		vocab_fn += '.' + postfix
		lowercase_vocab_fn += '.' + postfix
	save_vocab(vocab_fn, vocab)
	save_vocab(lowercase_vocab_fn, lowercase_vocab)

def main(argv):
	try:
		argv = FLAGS(argv)  # parse flags
	except gflags.FlagsError as e:
		print('%s\\nUsage: %s ARGS\\n%s' % (e, sys.argv[0], FLAGS))
		sys.exit(1)

	vocab, lowercase_vocab = gen_vocab(FLAGS.data_dir)
	output_dir = FLAGS.output_dir
	if output_dir == "":
		output_dir = os.path.join(FLAGS.data_dir, "vocab")
	save_vocabs(output_dir, vocab, lowercase_vocab, "")
	filter_vocab(vocab, 50)
	filter_vocab(lowercase_vocab, 50)
	save_vocabs(output_dir, vocab, lowercase_vocab, "50")
	filter_vocab(vocab, 100)
	filter_vocab(lowercase_vocab, 100)
	save_vocabs(output_dir, vocab, lowercase_vocab, "100")
	filter_vocab(vocab, 200)
	filter_vocab(lowercase_vocab, 200)
	save_vocabs(output_dir, vocab, lowercase_vocab, "200")
	filter_vocab(vocab, 500)
	filter_vocab(lowercase_vocab, 500)
	save_vocabs(output_dir, vocab, lowercase_vocab, "500")
	filter_vocab(vocab, 1000)
	filter_vocab(lowercase_vocab, 1000)
	save_vocabs(output_dir, vocab, lowercase_vocab, "1000")


if __name__ == '__main__':
	main(sys.argv)
