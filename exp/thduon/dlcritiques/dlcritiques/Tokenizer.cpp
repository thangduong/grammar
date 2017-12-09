#include "Tokenizer.h"
#include "BorrowedTokenizer.h"
Tokenizer::Tokenizer() {
}

Tokenizer::~Tokenizer() {
}

int Tokenizer::TokenizeText(
	const string& text,
	list<string>* tokens,
	list<int>* token_positions
) {
	// TODO: rewrite this function!
	BorrowedCode::Tokenizer t;
	t.set(text);
	string tok;
	int result = 0;
	while ("" != (tok = t.next())) {
		tokens->push_back(tok);
		result++;
	}
	return result;
}
