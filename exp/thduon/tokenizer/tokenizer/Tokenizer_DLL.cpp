#include <Windows.h>
#include <vector>
#include <algorithm>
#include <iostream>
#include <string>
#include <memory>
using namespace std;

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
