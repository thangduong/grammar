#include "Tokenizer_wrapper.h"
#include "Tokenizer.h"


struct TokenizerObject {
	Tokenizer tokenizer;
};

struct TokenizerResult {
	list<tuple<int, int, int>> _token_descs;
	list<string> _tokens;

	list<tuple<int, int, int>>::iterator _token_desc_itr;
	list<string>::iterator _token_itr;
};
#ifdef _WIN32
#define DLL_EXPORT	__declspec(dllexport)
#else
#define DLL_EXPORT
#endif

extern "C" {
	DLL_EXPORT void* __stdcall LoadTokenizer() {
		return new TokenizerObject();
	}
	DLL_EXPORT void UnloadTokenizer(void* tokenizer) {
		delete reinterpret_cast<TokenizerObject*>(tokenizer);
	}
	DLL_EXPORT void* __stdcall TokenizeString(void* tokenizer, const char* input_string, bool translit) {
		TokenizerResult* pResult = new TokenizerResult;
		TokenizerObject* pTokenizer = reinterpret_cast<TokenizerObject*>(tokenizer);
		pResult->_token_descs = pTokenizer->tokenizer.Tokenize(input_string, translit, &pResult->_tokens);
		pResult->_token_itr = pResult->_tokens.begin();
		pResult->_token_desc_itr = pResult->_token_descs.begin();
		return pResult;
	}
	DLL_EXPORT const char* __stdcall  NextToken(void* result, int* start, int* len, int* type) {
		TokenizerResult* pResult = reinterpret_cast<TokenizerResult*>(result);
		if (pResult->_token_itr == pResult->_tokens.end())
			return 0;
		const char* result_token = pResult->_token_itr->c_str();
		if (start)
			(*start) = get<0>(*pResult->_token_desc_itr);
		if (len)
			(*len) = get<1>(*pResult->_token_desc_itr);
		if (type)
			(*type) = get<2>(*pResult->_token_desc_itr);
		pResult->_token_itr++;
		pResult->_token_desc_itr++;
		return result_token;
	}
	DLL_EXPORT void __stdcall FreeTokenizedResult(void* result) {
		delete reinterpret_cast<TokenizerResult*>(result);
	}

}