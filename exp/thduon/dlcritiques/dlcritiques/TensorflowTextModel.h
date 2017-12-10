#pragma once
#include "TensorflowFrameworkModel.h"
#include "Vocabulary.h"
class TensorflowTextModel :
	public TensorflowFrameworkModel
{
public:
	TensorflowTextModel();
	virtual ~TensorflowTextModel();

	virtual bool Load(const string& params_json_filepath, const string& graphdef_filepath, const string& vocab_json_filepath);

protected:
	Vocabulary _vocab;
};

