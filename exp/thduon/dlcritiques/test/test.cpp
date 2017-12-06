#include <Windows.h>
#include <stdint.h>

#pragma pack(push,1)
struct ReplacementCritique {
	int32_t startCharIndex;
	int32_t length;
	char targetString[128];
	char critiqueName[128];
};
#pragma pack(pop)

typedef int (__stdcall * InitFcn)();
typedef int (__stdcall * ReleaseFcn)();
typedef int (__stdcall * GenerateReplacementCritiquesFcn)(wchar_t*, ReplacementCritique*, int);
typedef int (__stdcall * GetFIBProbabilitiesFcn)(wchar_t*, int, int, wchar_t**, int, float*);

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