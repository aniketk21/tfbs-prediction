import sys
f = open(sys.argv[1])
l = f.readlines()
del l[0]
k = set()
for el in l:
    k.add(el)

res = ''
for el in k:
    res += el

w = open(sys.argv[1]+'_uniq.csv', 'w')
w.write(res)
w.close()
f.close()
