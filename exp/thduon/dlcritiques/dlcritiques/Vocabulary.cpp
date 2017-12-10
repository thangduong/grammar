#include "Vocabulary.h"
#include "json/json.h"
#include <fstream>
#include <Windows.h>

Vocabulary::Vocabulary()
{
}


Vocabulary::~Vocabulary()
{
}

static void DebugOut(const string& msg) {
	wstring wmsg;
	wmsg.assign(msg.begin(), msg.end());
	OutputDebugString(wmsg.c_str());
	OutputDebugString(L"\r\n");
}

bool Vocabulary::LoadJsonFile(const string& jsonFilename) {
	Json::Value val;
	Json::Reader reader;
	ifstream jsonFile(jsonFilename, ios_base::in);
	if (jsonFile.is_open() && reader.parse(jsonFile, val)) {
		Json::ValueType rootType = val.type();
		if (rootType == Json::ValueType::arrayValue) {
			_index_to_word.resize(val.size());
			for (Json::Value::ArrayIndex i = 0; i != val.size(); i++) {
				_word_to_index[val[i].asString()] = i;
				_index_to_word[i] = val[i].asString();
				//DebugOut(word.key().asString());
			}
		}
		else if (rootType == Json::ValueType::objectValue) {
			_index_to_word.resize(val.size());
			int index;
			string wordstr;
			for (Json::ValueIterator word = val.begin(); word != val.end(); word++) {
				index = (*word).asInt();
				wordstr = word.key().asString();
				_word_to_index[wordstr] = index;
				_index_to_word[index] = wordstr;
			}
		}
		DebugOut(string("Loaded : ") + jsonFilename);
		// parse!
		return true;
	}
	else {
		DebugOut(reader.getFormattedErrorMessages());
		return false;
	}
}

list<int> Vocabulary::IndexTokens(const list<string>& tokens) {
	list<int> result;
	for (auto token_itr = tokens.begin(); token_itr != tokens.end(); token_itr++) {
		if (_word_to_index.count(*token_itr))
			result.push_back(_word_to_index[*token_itr]);
		else	// TODO: make "unk" variable
			result.push_back(_word_to_index["unk"]);
	}
	return result;
}

int Vocabulary::IndexToken(const string& token) {
	if (_word_to_index.count(token))
		return _word_to_index[token];
	else if (_word_to_index.count(token))
		return _word_to_index["unk"];
	else
		return -1;
}
