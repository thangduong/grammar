#pragma once
#include "TensorflowTextModel.h"
#include "Vocabulary.h"
class TensorflowLanguageModel :
	public TensorflowTextModel
{
public:
	TensorflowLanguageModel();
	virtual ~TensorflowLanguageModel();

	virtual bool Load(const string& params_json_filepath,
		const string& graphdef_filepath,
		const string& vocab_json_filepath,
		const string& output_vocab_json_filepath);
protected:
	Vocabulary _output_vocab;
};

