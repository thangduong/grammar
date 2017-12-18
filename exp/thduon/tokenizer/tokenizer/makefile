CC = g++
CFLAGS = -O3 -std=c++11 -I../libtokenizer -fPIC
LFLAGS = -L../libtokenizer -ltokenizer

tokenizer_so.so: Tokenizer_wrapper.o
	$(CC) -o tokenizer_so.so -shared Tokenizer_wrapper.o $(LFLAGS)

Tokenizer_wrapper.o: Tokenizer_wrapper.cpp
	$(CC) -c Tokenizer_wrapper.cpp $(CFLAGS)

clean:
	-rm tokenizer_so.so
	-rm *.o