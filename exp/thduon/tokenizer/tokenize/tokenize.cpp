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
	size_t max_line_len = 0;
	ofstream tok_time_file("toktime.txt");
	while (getline(infile, line)) {
		auto t1 = Clock::now();
		number_of_lines_processed += 1;
		if ((line.find_first_of("<b>") != string::npos)
		 ||(line.length()>2000)
		 ){
			continue;
		}
		string tokenized_line = tokenizer.TokenizeAndJuxtapose(line);
		auto t2 = Clock::now();
		auto read_time = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
		auto tok_time = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
		tok_time_file << tok_time << " " << line.length() << endl;
		if ((tok_time > 20)|| (tok_time > max_tok_time)) {
			cerr << line << endl;
			cerr << tokenized_line << endl;
			cerr << "tok_time = " << tok_time << endl;
			if (tok_time > max_tok_time) {
				max_tok_time = tok_time;
				cerr << line << endl;
				cerr << tokenized_line << endl;
				cerr << "max_tok_time = " << max_tok_time << endl;
			}
			if (tok_time > 300) {
				cerr << "DROPPING THIS LINE FROM THE DATA" << endl;
				continue;
			}
		}

		cout  << tokenized_line << endl;
		if ((number_of_lines_processed % 10000) == 0) {
			tok_time_file.flush();
			cerr << "processed " << number_of_lines_processed << " lines" << endl;
		}
		t0 = Clock::now();
		write_time = std::chrono::duration_cast<std::chrono::milliseconds>(t0 - t2).count();

		if (read_time > max_read_time) {
			max_read_time = read_time;
			cerr << line << endl;
			cerr << tokenized_line << endl;
			cerr << "max_read_time = " << max_read_time << endl;
			cerr << "tok_time = " << tok_time << endl;
		}
		if (write_time > max_write_time) {
			max_write_time = write_time;
			cerr << line << endl;
			cerr << tokenized_line << endl;
			cerr << "max_write_time = " << max_write_time << endl;
			cerr << "tok_time = " << tok_time << endl;
		}
		if (line.length() > max_line_len) {
			max_line_len = line.length();
			cerr << line << endl;
			cerr << tokenized_line << endl;
			cerr << "max_line_len = " << max_line_len << endl;
			cerr << "tok_time = " << tok_time << endl;
		}
	}
	return 0;
}