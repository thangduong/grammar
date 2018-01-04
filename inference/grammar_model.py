from tokenex.tokenizer import Tokenizer
class GrammarModel:
	def __init__(self, tokenizer=None):
		"""
		Construct grammar model.
		@param tokenizer: tokenizer object to use.  This must have
		a function tokenize_ex(.).
		"""
		if tokenizer is None:
			tokenizer = Tokenizer(0)
		self._tokenizer = tokenizer

	def critique(self, sentence, conf=0.5):
		None

