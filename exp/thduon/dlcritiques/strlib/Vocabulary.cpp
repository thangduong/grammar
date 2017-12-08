#include "Vocabulary.h"
#include "json/json.h"
#include <fstream>

Vocabulary::Vocabulary()
{
}


Vocabulary::~Vocabulary()
{
}

bool Vocabulary::LoadJsonFile(const char* jsonFilename) {
	Json::Value val;
	Json::Reader reader;
	std::ifstream jsonFile(jsonFilename);
	if (reader.parse(jsonFile, val)) {
		// parse!
		return true;
	} else
		return false;
}