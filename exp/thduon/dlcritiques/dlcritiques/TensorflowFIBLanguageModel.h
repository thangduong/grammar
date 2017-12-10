#pragma once
#include "TensorflowLanguageModel.h"
#include "Tokenizer.h"
#include <list>

using namespace std;

class TensorflowFIBLanguageModel :
	public TensorflowLanguageModel
{

public:
	TensorflowFIBLanguageModel();
	virtual ~TensorflowFIBLanguageModel();

	virtual bool Load(const string& params_json_filepath,
		const string& graphdef_filepath,
		const string& vocab_json_filepath,
		const string& output_vocab_json_filepath);
};

