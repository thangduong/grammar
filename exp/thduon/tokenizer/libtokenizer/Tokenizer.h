#pragma once
#include <string>
#include <set>
#include <unordered_map>
#include <regex>
#include <iterator>
#include <iostream>
using namespace std;

class Tokenizer
{
	enum {
		REGEX_TOKEN_TYPE_START = 100
	};
	list<string> _discard_delimiters;
	list<string> _retain_delimiters;
	list<string> _exception_tokens;
	unordered_map<string, string> _translit_map;
	int _min_translit_len;
	int _max_translit_len;
	vector<tuple<regex, string, bool>> _exception_token_group_regex;

	inline size_t ExactStringMatch(const list<string>& candidates, const string& input_string, int start)
	{
		for (auto candidate_itr = candidates.begin();
			candidate_itr != candidates.end(); candidate_itr++) {
			string candidate = *candidate_itr;
			size_t len = candidate.length();
			if (input_string.compare(start, len, candidate) == 0) {
				return candidate_itr->length();
			}
		}
		return 0;
	}
	inline size_t RegexStringMatch(vector<tuple<regex,string,bool>>& candidates, const string& input_string, int start, int* pattern_matched)
	{
		int idx = 0;
		for (auto candidate_itr = candidates.begin();
			candidate_itr != candidates.end(); candidate_itr++, idx++) {
			smatch sm;
			regex_search(next(input_string.begin(), start), input_string.end(), sm, get<0>(*candidate_itr));
			if (sm.length()) {
				if (pattern_matched)
					(*pattern_matched) = idx + REGEX_TOKEN_TYPE_START;
				size_t result = sm[0].length();
				return result;
			}
		}
		
		return 0;
	}

	bool CheckDelimiters(int matched_token_len,
		int matched_type,
		const string& input_string,
		int& start_marker,
		int& marker,
		list<tuple<int, int, int>>& start_len_type,
		bool translit,
		list<string>* token_list);

	string Tokenizer::Translit(const string& input_string, bool translit) {
		if (translit)
			return Translit(input_string);
		else
			return input_string;
	}
public:
	Tokenizer();
	virtual ~Tokenizer();

	string Translit(const string& input_string);
	list<tuple<int, int, int>> Tokenize(const string& input_string, bool translit, list<string>* token_list = 0);
	string TokenizeAndJuxtapose(const string& input_string, bool translit = true, list<pair<int, int>>* token_start_len = 0);
	
};

