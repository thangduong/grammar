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
		cout << "Usage: " << argv[0] << " <test_text>" << endl;
	}
	Tokenizer tokenizer;
	string tokenized_line = tokenizer.TokenizeAndJuxtapose(argv[1]);
	cout <<  "[" << tokenized_line << "]" << endl;
	return 0;
}