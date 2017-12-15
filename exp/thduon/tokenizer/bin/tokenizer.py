from ctypes import *
import os

script_dir = os.path.dirname(os.path.realpath(__file__))

if os.name=='nt':
	tok_so = cdll.LoadLibrary(os.path.join(script_dir,'tokenizer_dll.dll'))
	tokenizer = tok_so.LoadTokenizer()
else:
	tok_so = cdll.LoadLibrary(os.path.join(script_dir,'tokenizer_so.so'))
	tokenizer = tok_so.LoadTokenizer()
"""
input_string = c_char_p("hello this is a 500 test!".encode('utf-8'))
result = tok_so.TokenizeString (tokenizer, input_string, True)
tok_so.NextToken.argtypes = [c_void_p, POINTER(c_int), POINTER(c_int), POINTER(c_int)]
start = c_int()
len = c_int()
type = c_int()
done = False
while (not done):
	ctoken = tok_so.NextToken(result, byref(start), byref(len), byref(type))
	done = ctoken == 0
	if not done:
		token = [c_char_p(ctoken).value.decode('utf-8'), start.value, len.value, type.value]
		print(token)
#tok_so.FreeTokenizedResult(result)
#tok_so.UnloadTokenizer(tokenizer)
"""
def tokenize_string(string, translit=True):
	input_string = c_char_p(string.encode('utf-8'))
	tokenizer_result = tok_so.TokenizeString(tokenizer, input_string, translit)
	tok_so.NextToken.argtypes = [c_void_p, POINTER(c_int), POINTER(c_int), POINTER(c_int)]
	start = c_int()
	len = c_int()
	type = c_int()
	done = False
	result = []
	while (not done):
		ctoken = tok_so.NextToken(tokenizer_result, byref(start), byref(len), byref(type))
		done = ctoken == 0
		if not done:
			token = [c_char_p(ctoken).value.decode('utf-8'), start.value, len.value, type.value]
			result.append(token)
	return result

#print(tokenize_string("this is a test 510-847-7898 i am 50 years old from the 90s"))