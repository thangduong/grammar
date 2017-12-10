#include <Windows.h>
#include <stdint.h>
#include "AuxiliaryProofingEngine.h"
#include <string>
#include <iostream>

using namespace std;

int main(int argc, char* argv[]) {
	HMODULE hDLC = LoadLibrary(L"dlcritiques.dll");
	PFN_GetFillInBlankProbs GetFillinBlankProbs = (PFN_GetFillInBlankProbs)GetProcAddress(hDLC, "GetFillInBlankProbs");
	PFN_Init Init = (PFN_Init)GetProcAddress(hDLC, "Init");
	Init();
	wchar_t* word_choices[] = {
		L"something",
		L"testing"
	};
	float prob[1 + sizeof(word_choices) / sizeof(word_choices[0])] = { 0.0f };
	GetFillinBlankProbs(L"Hello world , how are you doing ?", 6, 5, 
		(const wchar_t**)word_choices, sizeof(word_choices)/sizeof(word_choices[0]), 
		prob);
	for (int i = 0; i < sizeof(prob) / sizeof(prob[0]); i++) {
		wstring wword = L"-";
		if (i>0)
			wword = word_choices[i-1];
		string word;
		word.assign(wword.begin(), wword.end());
		cout << word << " : " << prob[i] << endl;
	}
	FreeLibrary(hDLC);
	return 0;
}