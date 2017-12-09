#pragma once
#include <list>
#include <string>
#include <unordered_map>

using namespace std;

class Tokenizer {
	list<string> _keep_separators;
	list<string> _discard_separators;
	list<string> _overide_words;
	unordered_map<char, char> _char_replacement;
public:
	Tokenizer();
	virtual ~Tokenizer();
	int TokenizeText(
		const string& text,
		list<string>* tokens,
		list<int>* token_positions = 0
	);
};