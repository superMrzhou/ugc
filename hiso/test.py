def iterdata():
    while 1:
        for i in range(10):
            yield {'a':i, 'b':i**2}
it = iterdata()
aa = [it.next() for i in range(10)]
print(aa)
