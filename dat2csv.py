'''
    dat2csv.py
    usage: python dat2csv.py dataset.dat
    convert the dat file to csv file so that it can fit in memory when opened in Weka
'''
import sys

f = open(sys.argv[1])
w = open(sys.argv[1]+'.csv', 'w')

l = f.readlines()

for i in range(len(l)):
    l[i] = l[i].split()
    del l[i][0]
    del l[i][0]

res = ''

for i  in range(len(l)):
    res += l[i][0]+','+l[i][1]+','+l[i][2]+','+l[i][3]+','
    if l[i][4] == '0':
        res += 'no\n'
    else:
        res += 'yes\n'

w.write('chrom,e2f4_score,e2f6_score,dnase_score,peak\n')
w.write(res)

w.close()
f.close()
