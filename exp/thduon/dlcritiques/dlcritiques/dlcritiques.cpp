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
using namespace std;
using namespace tensorflow;

const string lm_graph_def_filename = "commaV5.graphdef";
std::shared_ptr<Session> lm_session;

BOOL WINAPI DllMain(_In_ HINSTANCE hinstDLL, _In_ DWORD     fdwReason, _In_ LPVOID    lpvReserved) {
	switch (fdwReason)
	{
	case DLL_THREAD_DETACH:
		// A thread exits normally.
		break;

	case DLL_PROCESS_DETACH:
		// A process unloads the DLL.
		break;
	}
	return TRUE;
}


extern "C"{ 
DLL_FUNCTION int __stdcall Init() {
	GraphDef graph_def;
	SessionOptions options;

	// load the LM graph
	ReadBinaryProto(Env::Default(), lm_graph_def_filename, &graph_def);
	Session* session = 0;
	Status status = NewSession(options, &session);
	lm_session = std::shared_ptr<Session>(session);
	TF_CHECK_OK(lm_session->Create(graph_def));


	return 0;
}

DLL_FUNCTION int __stdcall Release() {
	lm_session->Close();
	return 0;
}

DLL_FUNCTION int __stdcall GenerateReplacementCritiques(wchar_t* sentence,
	ReplacementCritique* critiquesOutBuffer,
	int bufferSize) {
	return 0;
}

DLL_FUNCTION int __stdcall GetFIBProbabilities(wchar_t* sentence,
	int blankStartIndex,
	int blankLen,
	wchar_t** choices,
	int numChoices,
	float* outBuffer) {
	return 0;
}

}