#include "TensorflowFIBLanguageModel.h"



TensorflowFIBLanguageModel::TensorflowFIBLanguageModel()
{
}


TensorflowFIBLanguageModel::~TensorflowFIBLanguageModel()
{
}

bool TensorflowFIBLanguageModel::Load(
	const string& params_json_filepath,
	const string& graphdef_filepath,
	const string& vocab_json_filepath,
	const string& output_vocab_json_filepath
	) {
	if (TensorflowLanguageModel::Load(params_json_filepath, graphdef_filepath, vocab_json_filepath, output_vocab_json_filepath)) {
		return true;
	} else
		return false;
}
