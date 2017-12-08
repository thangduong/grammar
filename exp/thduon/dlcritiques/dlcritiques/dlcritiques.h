#pragma once
#include <Windows.h>
#include <stdint.h>
#include "AuxiliaryProofingEngine.h"

#ifdef DLL_BUILD
#define	DLL_FUNCTION __declspec(dllexport)
#else
#define	DLL_FUNCTION __declspec(dllimport)
#endif

/*

#pragma pack(push,1)
struct ReplacementCritique {
	int32_t startCharIndex;
	int32_t length;
	char targetString[128];
	char critiqueName[128];
};
#pragma pack(pop)

extern "C" {
	DLL_FUNCTION int __stdcall GenerateReplacementCritiques(wchar_t* sentence,
		ReplacementCritique* critiquesOutBuffer,
		int bufferSize);
	DLL_FUNCTION int __stdcall GetFIBProbabilities(wchar_t* sentence,
		int blankStartIndex,
		int blankLen,
		wchar_t** choices,
		int numChoices,
		float* outBuffer);
}
*/