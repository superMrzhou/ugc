def test():
    return 1,2,3

b = [test() for i in range(3)]

print(b)
print(zip(*b))
