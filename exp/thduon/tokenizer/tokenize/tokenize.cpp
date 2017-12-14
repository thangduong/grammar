#include <iostream>
#include <fstream>
#include <string>
#include "Tokenizer.h"

using namespace std;

int main(int argc, char* argv[]) {
	if (argc != 2) {
		cout << "Usage: " << argv[0] << " <input-file>" << endl;
		cout << "output goes to stdout" << endl;
	}
	Tokenizer tokenizer;
	ifstream infile(argv[1]);
	string line;
	int number_of_lines_processed = 0;
	while (getline(infile, line)) {
		string tokenized_line = tokenizer.TokenizeAndJuxtapose(line);
		cout << tokenized_line << endl;
		number_of_lines_processed += 1;
		if ((number_of_lines_processed % 10000) == 0) {
			cerr << "processed " << number_of_lines_processed << " lines" << endl;
		}
	}
	return 0;
}