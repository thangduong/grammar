#pragma once
#include "TensorflowFIBLanguageModel.h"
#include <string>

using namespace std;

class TensorflowConvFIBLM :
	public TensorflowFIBLanguageModel
{
public:
	TensorflowConvFIBLM();
	virtual ~TensorflowConvFIBLM();


	virtual list<float> Eval(const string& before, const string& after, const list<string>& word_list);

	virtual bool Load(const string& params_json_filepath,
		const string& graphdef_filepath,
		const string& vocab_json_filepath,
		const string& output_vocab_json_filepath);

protected:
	int _num_words_before;
	int _num_words_after;
	string _start_token;
	bool _all_lowercase;
};

