#pragma once
#define COMPILER_MSVC
#define NOMINMAX
#include <Windows.h>
#include <vector>
#include <algorithm>
#include <iostream>
#include <string>
#include <memory>
#include <eigen/Dense>
#include "dlcritiques.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"
#include "Vocabulary.h"
#include "TensorflowConvFIBLM.h"
using namespace std;
using namespace tensorflow;

volatile short _ref_count = 0;
shared_ptr<TensorflowConvFIBLM> _tf_lm;
const string _model_name = "clmtV1";
const string _model_dir = _model_name + "\\";
const string _lm_param_filepath = _model_dir + "params.json";
const string _lm_graph_def_filepath = _model_dir+_model_name+".graphdef";
const string _lm_in_vocab_filepath = _model_dir + "vocab.json";
const string _lm_out_vocab_filepath = _model_dir + "keywords.json";

BOOL WINAPI DllMain(_In_ HINSTANCE hinstDLL, _In_ DWORD     fdwReason, _In_ LPVOID    lpvReserved) {
	switch (fdwReason)
	{
	case DLL_PROCESS_ATTACH:
		break;

	case DLL_PROCESS_DETACH:
		break;
	}
	return TRUE;
}


extern "C"{ 

__declspec(dllexport) int __stdcall Init() {
	if (InterlockedIncrement16(&_ref_count) == 1) {
		_tf_lm = shared_ptr<TensorflowConvFIBLM>(new TensorflowConvFIBLM());
		_tf_lm->Load(_lm_param_filepath, _lm_graph_def_filepath, _lm_in_vocab_filepath, _lm_out_vocab_filepath);
	}
	return 0;
}

__declspec(dllexport) uint32_t __stdcall GenerateCritiques(__in_z const wchar_t* wzInput,
	__out_ecount(cCritiquesOut) AuxProofingCritique* rgCritiquesOut,
	uint32_t cCritiquesOut) {
	return 0;
}

__declspec(dllexport) uint32_t __stdcall GetFillInBlankProbs(
	__in_z const wchar_t* wzSentence,
	uint32_t iStartBlank,
	uint32_t cchBlank,
	__in_ecount(cChoices) const wchar_t** rgChoices,
	uint32_t cChoices,
	__out_ecount(cChoices + 1) float* rgProbsOut) {
	wstring input_wstr = wzSentence;
	string input_str;
	input_str.assign(input_wstr.begin(), input_wstr.end());

	list<string> word_list;
	word_list.push_back(input_str.substr(iStartBlank, cchBlank));
	for (int i = 0; i < cChoices; i++) {
		wstring wchoice = rgChoices[i];
		string choice;
		choice.assign(wchoice.begin(), wchoice.end());
		word_list.push_back(choice);
	}
	list<float> prob_list = _tf_lm->Eval(input_str.substr(0, iStartBlank),
		input_str.substr(iStartBlank + cchBlank),
		word_list);
	assert(rgProbsOut != 0);
	for (auto prob_itr = prob_list.begin();
		prob_itr != prob_list.end(); prob_itr++) {
		(*rgProbsOut) = *prob_itr;
		rgProbsOut++;
	}
	return 0;
}

}