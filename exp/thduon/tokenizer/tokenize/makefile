CC = g++
CFLAGS = -O3 -std=c++11 -I../libtokenizer
LFLAGS = -L../libtokenizer -ltokenizer


all: tokenize test

tokenize: tokenize.cpp
	$(CC) -o tokenize tokenize.cpp $(CFLAGS) $(LFLAGS)

test: test.cpp
	$(CC) -o test test.cpp $(CFLAGS) $(LFLAGS)

clean:
	-rm test
	-rm tokenize
	-rm *.o