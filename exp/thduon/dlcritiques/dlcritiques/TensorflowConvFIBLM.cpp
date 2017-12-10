#include "TensorflowConvFIBLM.h"
#include <algorithm>
#include "Tokenizer.h"

TensorflowConvFIBLM::TensorflowConvFIBLM()
{
}


TensorflowConvFIBLM::~TensorflowConvFIBLM()
{
}

list<float> TensorflowConvFIBLM::Eval(const string& before, const string& after, const list<string>& word_list) {
	Tokenizer tokenizer;
	string before_str = _start_token + " " + before;
	string after_str = after;
	if (_all_lowercase) {
		transform(before_str.begin(), before_str.end(), before_str.begin(), ::tolower);
		transform(after_str.begin(), after_str.end(), after_str.begin(), ::tolower);
	}
	list<string> before_tokens, after_tokens;
	tokenizer.TokenizeText(before_str, &before_tokens);
	tokenizer.TokenizeText(after_str, &after_tokens);
	while (before_tokens.size() < _num_words_before)
		before_tokens.insert(before_tokens.begin(), "<pad>");
	while (after_tokens.size() < _num_words_after)
		after_tokens.push_back("<pad>");
	list<int> indexed_before_tokens = _vocab.IndexTokens(before_tokens);
	list<int> indexed_after_tokens = _vocab.IndexTokens(after_tokens);
	int sentence_len = _num_words_before + _num_words_after;

	// convert to tensors
//	vector<int32_t> sentence(indexed_before_tokens.size() + indexed_after_tokens.size());
	Tensor sentence_tensor(DT_INT32, TensorShape({ 1, sentence_len }));
	auto sentence_data = sentence_tensor.tensor<int32_t, 2>();
	uint32_t* sentence = (uint32_t*)sentence_data.data();
	//	memcpy(sentence_data.data(), &sentence[0], sentence.size() * sizeof(sentence[0]));
	int i = 0;
	for (auto tok_idx = indexed_before_tokens.begin();
		tok_idx != indexed_before_tokens.end();
		tok_idx++) {
		sentence[i++] = *tok_idx;
	}
	for (auto tok_idx = indexed_after_tokens.begin();
		tok_idx != indexed_after_tokens.end();
		tok_idx++) {
		sentence[i++] = *tok_idx;
	}

	Tensor is_training = Tensor(DataType::DT_BOOL, TensorShape({ 1, 1 }));

	vector<Tensor> outputs;
	vector<pair<string, Tensor>> inputs = {
		{ "sentence", sentence_tensor } };

	DWORD before_time = timeGetTime();
	Status status = _tf_session->Run(inputs, { "sm_decision" }, {}, &outputs);
	DWORD exec_time = timeGetTime() - before_time;
	int d0 = outputs[0].dim_size(0);
	int d1 = outputs[0].dim_size(1);
	char tmp[1024];
	sprintf(tmp, "d0=%d d1=%d exec time = %d", d0, d1, exec_time);
	DebugOut(tmp);
	
	float* sm_decision = (float*)outputs[0].tensor_data().data();
	float prob;
	list<float> result;
	for (list<string>::const_iterator word_itr = word_list.begin(); word_itr != word_list.end(); word_itr++) {
		char msg[1024];
		string word = *word_itr;
		if (_all_lowercase) 
			transform(word.begin(), word.end(), word.begin(), ::tolower);
		int word_idx = _vocab.IndexToken(word);
		if (word_idx >= 0) {
			prob = sm_decision[word_idx];
		}
		else {
			prob = -1.0;
		}
		result.push_back(prob);
		sprintf(msg, "%s - %d - %0.8f", word_itr->c_str(), word_idx, prob);
		DebugOut(msg);
	}
	return result;
}

bool TensorflowConvFIBLM::Load(const string& params_json_filepath,
	const string& graphdef_filepath,
	const string& vocab_json_filepath,
	const string& output_vocab_json_filepath) {
	if (TensorflowFIBLanguageModel::Load(params_json_filepath,
		graphdef_filepath,
		vocab_json_filepath,
		output_vocab_json_filepath)) {
		_num_words_before = _params["num_words_before"].asInt();
		_num_words_after = _params["num_words_after"].asInt();
		_all_lowercase = _params["all_lowercase"].asBool();
		_start_token = _params["start_token"].asString();

		// NOTE: this isn't necessary since it gets lowered
		// in the text after appending, but just in case!
		if (_all_lowercase)
			transform(_start_token.begin(), _start_token.end(), _start_token.begin(), ::tolower);
		return true;
	}
	else {
		return false;
	}
}