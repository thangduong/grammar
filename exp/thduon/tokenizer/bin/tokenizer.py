from ctypes import *
import os

script_dir = os.path.dirname(os.path.realpath(__file__))

if os.name=='nt':
	tok_so = cdll.LoadLibrary(os.path.join(script_dir,'tokenizer.dll'))
else:
	tok_so = cdll.LoadLibrary(os.path.join(script_dir,'tokenizer_so.so'))
tok_so.LoadTokenizer.argtypes = []
tok_so.LoadTokenizer.restype = c_void_p
tok_so.TokenizeString.argtypes = [c_void_p, c_char_p, c_bool]
tok_so.TokenizeString.restype = c_void_p
tok_so.NextToken.argtypes = [c_void_p, POINTER(c_int), POINTER(c_int), POINTER(c_int)]
tok_so.NextToken.restype = c_char_p
tok_so.FreeTokenizedResult.argtypes = [c_void_p]

class Tokenizer:
	def __init__(self):
		self._tokenizer = tok_so.LoadTokenizer()

	def tokenize(self, string, translit=True):
		input_string = c_char_p(string.encode('utf-8'))
		tokenizer_result = tok_so.TokenizeString(self._tokenizer, input_string, translit)
		start = c_int()
		len = c_int()
		type = c_int()
		done = False
		result = []
		while (not done):
			ctoken = tok_so.NextToken(tokenizer_result, byref(start), byref(len), byref(type))
			if ctoken is not None:
				token = [c_char_p(ctoken).value.decode('utf-8'), start.value, len.value, type.value]
				result.append(token)
			else:
				done = True
		tok_so.FreeTokenizedResult(tokenizer_result)
		return result
