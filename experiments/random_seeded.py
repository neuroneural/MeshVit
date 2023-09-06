import random
import sys

if len(sys.argv) < 3:
    print("Usage: python random_seeded.py <seed> <list of values>")
    sys.exit(1)

seed = int(sys.argv[1])
values = sys.argv[2:]

random.seed(seed)
print(random.choice(values))
