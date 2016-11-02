import sys
from collections import defaultdict

words_dict = defaultdict(int)

for line in sys.stdin:
    words = line.rstrip("\n").split()
    for word in words:
        words_dict[word] += 1

print(len(words_dict.keys()))

