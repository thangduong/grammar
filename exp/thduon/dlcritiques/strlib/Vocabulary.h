#pragma once
#include <unordered_map>

using namespace std;

class Vocabulary
{
	// map from vocabulary to its index
	unordered_map<string, int> _word_to_index;
	vector<string> _index_to_word;
public:
	Vocabulary();
	virtual ~Vocabulary();
	bool LoadJsonFile(const char* json_filename);
};

