all: libtokenizer/libtokenizer.a tokenizer/tokenizer_so.so tokenize/tokenize
	-mkdir bin
	cp libtokenizer/libtokenizer.a tokenizer/tokenizer_so.so tokenize/tokenize tokenize/test bin
	echo all done!
clean:
	-rm bin/libtokenizer.a bin/tokenizer_so.so bin/tokenize
	-rm libtokenizer/libtokenizer.a tokenizer/tokenizer_so.so tokenize/tokenize
	make -C libtokenizer clean
	make -C tokenizer clean
	make -C tokenize clean
tokenizer/tokenizer_so.so: libtokenizer/libtokenizer.a
	make -C tokenizer
tokenize/tokenize: libtokenizer/libtokenizer.a tokenize/tokenize.cpp tokenize/makefile
	make -C tokenize
libtokenizer/libtokenizer.a: libtokenizer/makefile libtokenizer/Tokenizer.cpp libtokenizer/Tokenizer.h
	make -C libtokenizer
