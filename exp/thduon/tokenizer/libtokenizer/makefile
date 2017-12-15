CC = g++
CFLAGS = -O3 -std=c++11 -fPIC

libtokenizer.a: Tokenizer.o
	ar rcs libtokenizer.a Tokenizer.o

Tokenizer.o: Tokenizer.h Tokenizer.cpp
	$(CC) -c Tokenizer.cpp $(CFLAGS)
