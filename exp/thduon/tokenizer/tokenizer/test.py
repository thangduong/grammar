import tokenizer
import time
before = time.time()
y = tokenizer.tokenize_string("It costs the Open Content Alliance as much as $30 to scan each book, a cost shared by the group’s "\
"members and benefactors, so there are obvious financial benefits to libraries of Google’s wide-ranging offer, started in 2004.",True)
diff = time.time() - before
print(y)
print(diff)
