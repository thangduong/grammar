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

const string lm_graph_def_filename = ".graphdef";
std::shared_ptr<Session> lm_session;
std::shared_ptr<Session> wrm_session;

int __stdcall Init() {
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

int __stdcall Release() {
	lm_session->Close();
	return 0;
}

BOOL WINAPI DllMain(_In_ HINSTANCE hinstDLL, _In_ DWORD     fdwReason, _In_ LPVOID    lpvReserved) {
	switch (fdwReason)
	{
	case DLL_PROCESS_ATTACH:
		Init();
		break;

	case DLL_PROCESS_DETACH:
		Release();
		break;
	}
	return TRUE;
}


extern "C"{ 

uint32_t __stdcall GenerateCritiques(__in_z const wchar_t* wzInput,
	__out_ecount(cCritiquesOut) AuxProofingCritique* rgCritiquesOut,
	uint32_t cCritiquesOut) {

}

uint32_t __stdcall GetFillInBlankProbs(__in_z const wchar_t* wzInput,
	__out_ecount(cCritiquesOut) AuxProofingCritique* rgCritiquesOut,
	uint32_t cCritiquesOut) {
	return 0;
}

}