///////////////////////////////////////////////////////////////////////////////
// Tokenizer.h
// ===========
// General purpose string tokenizer (C++ string version)
//
// The default delimiters are space(" "), tab(\t, \v), newline(\n),
// carriage return(\r), and form feed(\f).
// If you want to use different delimiters, then use setDelimiter() to override
// the delimiters. Note that the delimiter string can hold multiple characters.
//
//  AUTHOR: Song Ho Ahn (song.ahn@gmail.com)
// CREATED: 2005-05-25
// UPDATED: 2011-03-08
///////////////////////////////////////////////////////////////////////////////

#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <string>
#include <vector>
using namespace std;

namespace BorrowedCode {
// default delimiter string (space, tab, newline, carriage return, form feed)
const string DEFAULT_DELIMITER = " \t\v\n\r\f";

class Tokenizer
{
public:
    // ctor/dtor
    Tokenizer();
    Tokenizer(const string& str, const string& delimiter=DEFAULT_DELIMITER);
    ~Tokenizer();

    // set string and delimiter
    void set(const string& str, const string& delimiter=DEFAULT_DELIMITER);
    void setString(const string& str);             // set source string only
    void setDelimiter(const string& delimiter);    // set delimiter string only

    string next();                                 // return the next token, return "" if it ends

    vector<string> split();                   // return array of tokens from current cursor

protected:


private:
    void skipDelimiter();                               // ignore leading delimiters
    bool isDelimiter(char c);                           // check if the current char is delimiter

    string buffer;                                 // input string
    string token;                                  // output string
    string delimiter;                              // delimiter string
    string::const_iterator currPos;                // string iterator pointing the current position

};
}
#endif // TOKENIZER_H
