#include "TensorflowLanguageModel.h"



TensorflowLanguageModel::TensorflowLanguageModel()
{
}


TensorflowLanguageModel::~TensorflowLanguageModel()
{
}

bool TensorflowLanguageModel::Load(const string& params_json_filepath,
	const string& graphdef_filepath,
	const string& vocab_json_filepath,
	const string& output_vocab_json_filepath) {
	return TensorflowTextModel::Load(params_json_filepath, graphdef_filepath, vocab_json_filepath) 
		&& _output_vocab.LoadJsonFile(output_vocab_json_filepath);
}