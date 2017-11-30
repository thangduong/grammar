def delete_one_char(word):
	result = []
	for i in range(len(word)):
		result.append(word[:i]+word[(i+1):])
	return result

def swap_one_char(word):
	result = []
	for i in range(len(word)):
		result.append(word[:i]+word[(i+1):])
	return result

def insert_one_char(word):
	result = []
	for i in range(len(word)):
		for ch in
		result.append(word[:i]+word[(i+1):])
	return result


def mangle(word):
	None

if __name__ == "__main__":
	d = delete_one_char('that')
	print(d)