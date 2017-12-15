#include <iostream>
#include <fstream>
#include <string>
#include <iostream>
#include <chrono>
#include <thread>
#include "Tokenizer.h"

using namespace std;
typedef std::chrono::high_resolution_clock Clock;

int main(int argc, char* argv[]) {
	if (argc != 2) {
		cout << "Usage: " << argv[0] << " <input-file>" << endl;
		cout << "output goes to stdout" << endl;
	}
	Tokenizer tokenizer;
	ifstream infile(argv[1]);
	string line;
	int number_of_lines_processed = 0;
	auto t0 = Clock::now();
	long long write_time;
	long long max_read_time = 0, max_write_time = 0, max_tok_time = 0;
	while (getline(infile, line)) {
		auto t1 = Clock::now();
		string tokenized_line = tokenizer.TokenizeAndJuxtapose(line);
		auto t2 = Clock::now();
		auto read_time = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
		auto tok_time = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
		cout  << tokenized_line << endl;
		number_of_lines_processed += 1;
		if ((number_of_lines_processed % 10000) == 0) {
			cerr << "processed " << number_of_lines_processed << " lines" << endl;
		}
		t0 = Clock::now();
		write_time = std::chrono::duration_cast<std::chrono::milliseconds>(t0 - t2).count();
		if (read_time > max_read_time) {
			max_read_time = read_time;
			cerr << "max_read_time = " << max_read_time << endl;
			cerr << line << endl;
			cerr << tokenized_line << endl;
		}
		if (tok_time > max_tok_time) {
			max_tok_time = tok_time;
			cerr << "max_tok_time = " << max_tok_time << endl;
			cerr << line << endl;
			cerr << tokenized_line << endl;
		}
		if (write_time > max_write_time) {
			max_write_time = write_time;
			cerr << "max_write_time = " << max_write_time << endl;
			cerr << line << endl;
			cerr << tokenized_line << endl;
		}
	}
	return 0;
}