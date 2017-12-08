#include <Windows.h>
#include <stdint.h>
#include "AuxiliaryProofingEngine.h"

int main(int argc, char* argv[]) {
	HMODULE hDLC = LoadLibrary(L"dlcritiques.dll");
	InitFcn Init = (InitFcn)GetProcAddress(hDLC, "Init");
	ReleaseFcn Release = (ReleaseFcn)GetProcAddress(hDLC, "Release");
	GenerateReplacementCritiquesFcn GenerateReplacementCritiques = (GenerateReplacementCritiquesFcn)GetProcAddress(hDLC, "GenerateReplacementCritiques");
	GetFIBProbabilitiesFcn GetFIBProbabilities = (GetFIBProbabilitiesFcn)GetProcAddress(hDLC, "GetFIBProbabilities");
	Init();
	Release();
	return 0;
}