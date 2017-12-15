#pragma once

using PFN_LoadTokenizer = void*(__stdcall *)();
using PFN_UnloadTokenizer = void(__stdcall*)(void* tokenizer);
using PFN_TokenizeString = void*(__stdcall *)(void* tokenizer, const char* czInput, bool translit);
using PFN_NextToken = void*(__stdcall *)(void* result, int* start, int* len, int* type);
using PFN_FreeTokenizeResult = void(__stdcall*)(void* result);