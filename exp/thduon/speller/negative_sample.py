def delete_one_char(word):
	result = []
	for i in range(len(word)):
		result.append(word[:i]+word[(i+1):])
	return result

def swap_nearby_pair(word):
	result = []
	for i in range(1,len(word)):
		result.append(word[0:(i-1)]+word[i]+word[i-1] + word[(i+1):])
	return result

def mangle(word):
	None

if __name__ == "__main__":
	d = swap_nearby_pair('from')
	print(d)