#include "Tokenizer.h"
#include <list>
#include <iostream>
using namespace std;

Tokenizer::Tokenizer()
{
	// TODO: use hash + set membership to speed up tokenization
	_discard_delimiters.push_back(" ");
	_discard_delimiters.push_back("\t");
	_discard_delimiters.push_back("\r");
	_discard_delimiters.push_back("\n");
	_discard_delimiters.push_back(u8"¬");	// not sure what this is.  probably just noise!

	_retain_delimiters.push_back("''");
	_retain_delimiters.push_back(u8"$");
	_retain_delimiters.push_back(u8"`");
	_retain_delimiters.push_back(u8":");
	_retain_delimiters.push_back(u8"!");
	_retain_delimiters.push_back(u8"%");
	_retain_delimiters.push_back(u8"^");
	_retain_delimiters.push_back(u8")");
	_retain_delimiters.push_back(u8"(");
	_retain_delimiters.push_back(u8"]");
	_retain_delimiters.push_back(u8"[");
	_retain_delimiters.push_back(u8"}");
	_retain_delimiters.push_back(u8"{");
	_retain_delimiters.push_back(u8"@");
	_retain_delimiters.push_back(u8"#");
	_retain_delimiters.push_back(u8"~");
	_retain_delimiters.push_back(u8"-");
	_retain_delimiters.push_back(u8".");
	_retain_delimiters.push_back(u8"\"");
	_retain_delimiters.push_back(u8"'");
	_retain_delimiters.push_back(u8";");
	_retain_delimiters.push_back(u8",");
	_retain_delimiters.push_back(u8"™");
	_retain_delimiters.push_back(u8"€");
	_retain_delimiters.push_back(u8"°");
	_retain_delimiters.push_back(u8"?");
	_retain_delimiters.push_back(u8"*");
	_retain_delimiters.push_back(u8"|");
	_retain_delimiters.push_back(u8"…");
	_retain_delimiters.push_back(u8"=");
	_retain_delimiters.push_back(u8"+");
	_retain_delimiters.push_back(u8"&");
	_retain_delimiters.push_back(u8"/");
	_retain_delimiters.push_back(u8"\\");
	_retain_delimiters.push_back(u8"—");
	_retain_delimiters.push_back(u8"’");
	_retain_delimiters.push_back(u8"“");
	_retain_delimiters.push_back(u8"”");
	_retain_delimiters.push_back(u8"‘");
	_retain_delimiters.push_back(u8"′");
	_retain_delimiters.push_back(u8"¶");

	_exception_tokens.push_back("...");
	_exception_tokens.push_back("....");
	_exception_tokens.push_back(u8"can't");
	_exception_tokens.push_back(u8"can’t");
	_exception_tokens.push_back(u8"Can't");
	_exception_tokens.push_back(u8"Can’t");
	_exception_tokens.push_back(u8"'s");
	_exception_tokens.push_back(u8"'S");
	_exception_tokens.push_back(u8"n't");
	_exception_tokens.push_back(u8"N't");
	_exception_tokens.push_back(u8"n'T");
	_exception_tokens.push_back(u8"N'T");
	_exception_tokens.push_back(u8"’s");
	_exception_tokens.push_back(u8"’S");
	_exception_tokens.push_back(u8"n’t");
	_exception_tokens.push_back(u8"N’t");
	_exception_tokens.push_back(u8"n’T");
	_exception_tokens.push_back(u8"N’T");
	_exception_tokens.push_back(u8"'m");
	_exception_tokens.push_back(u8"'M");
	_exception_tokens.push_back(u8"’m");
	_exception_tokens.push_back(u8"’M");
	_exception_tokens.push_back("dr.");
	_exception_tokens.push_back("Dr.");
	_exception_tokens.push_back("no.");
	_exception_tokens.push_back("No.");
	_exception_tokens.push_back("mr.");
	_exception_tokens.push_back("Mr.");
	_exception_tokens.push_back("Ms.");
	_exception_tokens.push_back("ms.");
	_exception_tokens.push_back("Mrs.");
	_exception_tokens.push_back("mrs.");
	_exception_tokens.push_back(u8"s'");
	_exception_tokens.push_back(u8"S'");
	_exception_tokens.push_back(u8"s’");
	_exception_tokens.push_back(u8"S’");
	_exception_tokens.push_back(u8"i.e.");
	_exception_tokens.push_back(u8"e.g.");
	_exception_tokens.push_back("m.p.h.");
	_exception_tokens.push_back("M.P.H.");
	_exception_tokens.push_back("M.p.h.");

	_exception_tokens.push_back(u8"gov’t");
	_exception_tokens.push_back(u8"Gov’t");
	_exception_tokens.push_back(u8"GOV’T");

	_exception_tokens.push_back(u8"'ve");
	_exception_tokens.push_back(u8"’ve");
	_exception_tokens.push_back(u8"'Ve");
	_exception_tokens.push_back(u8"’Ve");
	_exception_tokens.push_back(u8"'vE");
	_exception_tokens.push_back(u8"’vE");
	_exception_tokens.push_back(u8"'VE");
	_exception_tokens.push_back(u8"’VE");

	_exception_tokens.push_back(u8"'re");
	_exception_tokens.push_back(u8"’re");
	_exception_tokens.push_back(u8"'Re");
	_exception_tokens.push_back(u8"’Re");
	_exception_tokens.push_back(u8"'rE");
	_exception_tokens.push_back(u8"’rE");
	_exception_tokens.push_back(u8"'RE");
	_exception_tokens.push_back(u8"’RE");

	_exception_tokens.push_back(u8"'ll");
	_exception_tokens.push_back(u8"’ll");
	_exception_tokens.push_back(u8"'Ll");
	_exception_tokens.push_back(u8"’Ll");
	_exception_tokens.push_back(u8"'lL");
	_exception_tokens.push_back(u8"’lL");
	_exception_tokens.push_back(u8"'LL");
	_exception_tokens.push_back(u8"’LL");

	_exception_tokens.push_back("(tm)");
	_exception_tokens.push_back("a.m.");
	_exception_tokens.push_back("p.m");
	_exception_tokens.push_back("jr.");
	_exception_tokens.push_back("Jr.");
	_exception_tokens.push_back("sr.");
	_exception_tokens.push_back("Sr.");
	_exception_tokens.push_back("hon.");
	_exception_tokens.push_back("hr.");
	_exception_tokens.push_back("hosp.");
	_exception_tokens.push_back("lt.");
	_exception_tokens.push_back("gen.");
	_exception_tokens.push_back("Gen.");
	_exception_tokens.push_back("e-mail");
	_exception_tokens.push_back("E-mail");
	_exception_tokens.push_back("E-Mail");
	_exception_tokens.push_back("Rock'n'Roll");
	_exception_tokens.push_back("Rock'N'Roll");
	_exception_tokens.push_back("rock'n'roll");
	_exception_tokens.push_back("rock'N'roll");
	_exception_tokens.push_back("'t");
	_exception_tokens.push_back(u8"`t");
//	_exception_tokens.push_back("t.v.");
//	_exception_tokens.push_back("r.p.m.");


	_translit_map[u8"™"] = "(tm)";
	_translit_map[u8"“"] = "\"";
	_translit_map[u8"”"] = "\"";
	_translit_map[u8"…"] = "...";
	_translit_map[u8"—"] = "--";
	_translit_map[u8"′"] = "'";
	_translit_map[u8"`"] = "'";
	_translit_map[u8"’"] = "'";
	_translit_map[u8"‘"] = "'";
	_translit_map[u8"…"] = "...";
	_translit_map["''"] = "\"";		// this is questionable
	_translit_map[u8"À"] = "A";
	_translit_map[u8"Á"] = "A";
	_translit_map[u8"Â"] = "A";
	_translit_map[u8"Ã"] = "A";
	_translit_map[u8"Ä"] = "A";
	_translit_map[u8"Å"] = "A";
	_translit_map[u8"è"] = "e";
	_translit_map[u8"é"] = "e";
	_translit_map[u8"ê"] = "e";
	_translit_map[u8"ë"] = "e";


	_max_translit_len = _min_translit_len = (int)_translit_map.begin()->first.length();
	for (auto translit_itr = _translit_map.begin(); translit_itr != _translit_map.end(); translit_itr++) {
		if ((int)translit_itr->first.length() < _min_translit_len)
			_min_translit_len = (int)translit_itr->first.length();
		else if ((int)translit_itr->first.length() > _max_translit_len)
			_max_translit_len = (int)translit_itr->first.length();
	}

	/*
	_translit_map[u8"™"] = "(tm)";
	_translit_map[u8"“"] = "\"";
	_translit_map[u8"”"] = "\"";
	_translit_map[u8"…"] = "...";
	_translit_map[u8"—"] = "--";
	_translit_map[u8"′"] = "'";
	_translit_map[u8"’"] = "'";
	_translit_map[u8"‘"] = "'";
	_translit_map[u8"’ve"] = "'ve";
	_translit_map[u8"’t"] = "'t";
	_translit_map[u8"’s"] = "'s";
	_translit_map[u8"’S"] = "'S";
	_translit_map[u8"n’t"] = "n't";
	_translit_map[u8"…"] = "...";
	_translit_map["...."] = "...";
	_translit_map["''"] = "\"";
	_translit_map[u8"’m"] = "'m";
	_translit_map[u8"’re"] = "'re";
	_translit_map[u8"’M"] = "'M";
	_translit_map[u8"s’"] = "s'";
	_translit_map[u8"S’"] = "S'";
	_translit_map[u8"can’t"] = "can't";
	*/

	// tuple meaning:
	//	0 = regular expression
	//	1 = string to identify token, if "", then string is actual matched string
	//	2 = true if this can act as a retain delimiter, false if not
	_exception_token_group_regex.push_back(tuple<regex, string, bool>(regex("^20[0-9]0s"), "<decade-year>", true));
	_exception_token_group_regex.push_back(tuple<regex, string, bool>(regex("^19[0-9]0s"), "<decade-year>", true));
	_exception_token_group_regex.push_back(tuple<regex, string, bool>(regex("^[0-9]0s"), "<decade>", true));
	_exception_token_group_regex.push_back(tuple<regex, string, bool>(regex("^\\([0-9][0-9][0-9]\\)[ -]*[0-9][0-9][0-9][ -]*[0-9][0-9][0-9][0-9]"), "<phone-number>", true));
	_exception_token_group_regex.push_back(tuple<regex, string, bool>(regex("^[0-9][0-9][0-9]-[0-9][0-9][0-9]-[0-9][0-9][0-9][0-9]"), "<phone-number>", true));
	//	_exception_token_group_regex.push_back(tuple<regex, string, bool>(regex("^19[0-9][0-9]"), "<year>"));
//	_exception_token_group_regex.push_back(tuple<regex, string, bool>(regex("^20[0-9][0-9]"), "<year>"));
	_exception_token_group_regex.push_back(tuple<regex, string, bool>(regex("^([0-9][0-9][0-9]|[0-9][0-9]|[0-9])(,[0-9][0-9][0-9])+"), "<large-number>", true));
	_exception_token_group_regex.push_back(tuple<regex, string, bool>(regex("^([0-9]*)[.]([0-9])+"), "<decimal>", true));
	_exception_token_group_regex.push_back(tuple<regex, string, bool>(regex("^([A-Za-z][.])+"), "", false));
	_exception_token_group_regex.push_back(tuple<regex, string, bool>(regex("^([4-9])([0-9])*"), "<integer>", true));
	_exception_token_group_regex.push_back(tuple<regex, string, bool>(regex("^0([0-9])+"), "<integer>", true));
	_exception_token_group_regex.push_back(tuple<regex, string, bool>(regex("^1([0-9])+"), "<integer>", true));
	_exception_token_group_regex.push_back(tuple<regex, string, bool>(regex("^2([0-9])+"), "<integer>", true));
	_exception_token_group_regex.push_back(tuple<regex, string, bool>(regex("^3([0-9])+"), "<integer>", true));
	_exception_token_group_regex.push_back(tuple<regex, string, bool>(regex("^0"), "<zero>", true));
	_exception_token_group_regex.push_back(tuple<regex, string, bool>(regex("^1"), "<one>", true));
	_exception_token_group_regex.push_back(tuple<regex, string, bool>(regex("^2"), "<two>", true));
	_exception_token_group_regex.push_back(tuple<regex, string, bool>(regex("^3"), "<three>", true));
	_exception_token_group_regex.push_back(tuple<regex, string, bool>(regex("^(\\w+)(\\.|_)?(\\w*)@(\\w+)(\\.(\\w+))+"), "<email-address>", true));
	_exception_token_group_regex.push_back(tuple<regex, string, bool>(regex("^http://[a-zA-z.]+(/[a-zA-z.]+)*"), "<url>", true));
	_exception_token_group_regex.push_back(tuple<regex, string, bool>(regex("^https://[a-zA-z.]+(/[a-zA-z.]+)*"), "<url>", true));
//	_exception_token_group_regex.push_back(tuple<regex, string, bool>(regex("^[a-z][.]([a-zA-Z][.])+"), "", true));


	// TODO:
	// ordinal numbers, 1st, 2nd, 3rd, 5th, 4th, 53rd, etc.
}


Tokenizer::~Tokenizer()
{
}
bool Tokenizer::CheckDelimiters(int matched_token_len, 
	int matched_type, 
	const string& input_string, 
	int& start_marker, 
	int& marker, 
	list<tuple<int, int, int>>& start_len_type, 
	bool translit,
	list<string>* token_list) {

	if (matched_token_len > 0) {
		int proposed_marker = marker + matched_token_len;
		if (
			   (ExactStringMatch(_discard_delimiters, input_string, proposed_marker) > 0) 
			|| (ExactStringMatch(_retain_delimiters, input_string, proposed_marker) > 0)
			|| (proposed_marker >= (int)input_string.length())
			){
			if (start_marker < marker) {
				tuple<int, int, int> token(start_marker, marker - start_marker, 0);
				start_len_type.push_back(token);
				if (token_list)
					token_list->push_back(Translit(input_string.substr(get<0>(token),get<1>(token)),translit));
			}
			tuple<int, int, int> token(marker, matched_token_len, matched_type);
			start_len_type.push_back(token);
			if (token_list) {
				if ((matched_type > 0) && (!get<1>(_exception_token_group_regex[matched_type]).empty()))
					token_list->push_back(Translit(get<1>(_exception_token_group_regex[matched_type]),translit));
				else
					token_list->push_back(Translit(input_string.substr(get<0>(token), get<1>(token)),translit));
			}
			marker = proposed_marker;
			start_marker = marker;
			return true;
		}
	}
	return false;
}


list<tuple<int,int,int>> Tokenizer::Tokenize(const string& input_string, bool translit, list<string>* token_list) {
	int start_marker = 0;
	int marker = 0;
	list<tuple<int, int, int>> result;

	while (marker < (int)input_string.length()) {
		
		size_t matched_exception_token_len = ExactStringMatch(_exception_tokens, input_string, marker);
		if ((matched_exception_token_len>0) 
			&& CheckDelimiters((int)matched_exception_token_len, 0, input_string, start_marker, marker, result, translit, token_list))
			continue;
		int regex_pattern_matched;
		size_t matched_regex_token_len = RegexStringMatch(_exception_token_group_regex, input_string, marker, &regex_pattern_matched);
		if ((matched_regex_token_len>0) 
			&& (get<2>(_exception_token_group_regex[regex_pattern_matched]) || (marker == start_marker))
			&& CheckDelimiters((int)matched_regex_token_len, regex_pattern_matched, input_string, start_marker, marker, result, translit, token_list))
			continue;

		size_t discard_token_len = ExactStringMatch(_discard_delimiters, input_string, marker);
		if (discard_token_len > 0) {
			if (start_marker < marker) {
				tuple<int, int, int> token(start_marker, marker - start_marker, 0);
				result.push_back(token);
				if (token_list)
					token_list->push_back(Translit(input_string.substr(get<0>(token), get<1>(token)),translit));
			}
			marker += 1;
			start_marker = marker;
			continue;
		}
		size_t retain_token_len = ExactStringMatch(_retain_delimiters, input_string, marker);
		if (retain_token_len > 0) {
			if (start_marker < marker) {
				tuple<int, int, int> token(start_marker, marker - start_marker, 0);
				result.push_back(token);
				if (token_list)
					token_list->push_back(Translit(input_string.substr(get<0>(token), get<1>(token)),translit));
			}
			tuple<int, int, int> token(marker, (int)retain_token_len, 0);
			result.push_back(token);
			if (token_list)
				token_list->push_back(Translit(input_string.substr(get<0>(token), get<1>(token)),translit));

			marker += (int)retain_token_len;
			start_marker = marker;
			continue;
		}
		marker += 1;
	} 
	if (start_marker < marker) {
		tuple<int, int, int> token(marker, marker - start_marker, 0);
		result.push_back(token);
		if (token_list)
			token_list->push_back(Translit(input_string.substr(get<0>(token), get<1>(token)),translit));
	}
	return result;
}

string Tokenizer::TokenizeAndJuxtapose(const string& input_string, bool translit, list<pair<int, int>>* token_start_len) {
	list<string> tokens;
	list<tuple<int, int, int>> token_start_len_type = Tokenize(input_string, translit, &tokens);
	string result;
	for (auto token = tokens.begin(); token != tokens.end(); token++) {
		if (!result.empty())
			result += " ";
		result += (*token);
	}
	return result;
}

string Tokenizer::Translit(const string& input_string) {
	int marker = 0;
	int start_marker = marker;
	string result = "";
	while (marker < input_string.length()) {
		for (auto len = _min_translit_len; len <= _max_translit_len; len++) {
			string piece = input_string.substr(marker, len);
			auto match = _translit_map.find(piece);
			if (match != _translit_map.end()) {
				result += input_string.substr(start_marker, marker - start_marker) + match->second;
				marker += len - 1;
				start_marker = marker + 1;
				break;
			}
		}
		marker += 1;
	}
	if (start_marker < input_string.length())
		result += input_string.substr(start_marker);
	return result;
}
