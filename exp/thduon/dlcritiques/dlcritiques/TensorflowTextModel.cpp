#include "TensorflowTextModel.h"



TensorflowTextModel::TensorflowTextModel()
{
}


TensorflowTextModel::~TensorflowTextModel()
{
}

bool TensorflowTextModel::Load(const string& params_json_filepath, const string& graphdef_filepath, const string& vocab_json_filepath) {
	if (TensorflowFrameworkModel::Load(params_json_filepath, graphdef_filepath) && _vocab.LoadJsonFile(vocab_json_filepath)) {
		return true;
	} else {
		return false;
	}
}
