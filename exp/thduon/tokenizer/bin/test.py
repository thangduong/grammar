import tokenizer as t
import sys
input_string = "hello, world siic's! on the 06th of this month and the 124th street, the 1st, 2nd, 3rd, 4th, 5th, the minor fall and the major lift"

if len(sys.argv)>1:
  input_string = sys.argv[1]
tok = t.Tokenizer()
tokens = tok.tokenize(input_string)

output = ""
for token in tokens:
  output += " %s"%token[0]

print(output)