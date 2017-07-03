'''
    mod_dnase_score.py
    usage: python mod_dnase_score.py bedgraph_file.bedgraph dataset.dat output.dat
    modify the DNASE score.
'''

import sys

bed = open(sys.argv[1])
inp = open(sys.argv[2])
out = open(sys.argv[3], 'w')

inpl = inp.readlines()
bedl = bed.readlines()

for i in xrange(len(bedl)):
    bedl[i] = bedl[i].split()

for i in xrange(len(inpl)):
    inpl[i] = inpl[i].split()

print('Length of ' + sys.argv[1] + ': ' + str(len(bedl)))
print('Length of ' + sys.argv[2] + ': ' + str(len(inpl)))

cnt = 0
i = 0
line = ''
for el in bedl:
    low = int(el[1])
    high = int(el[2])
    dnase = el[3]
    while True:
        elem = inpl[i]
        if int(elem[1]) <= high:
            line += str(elem[0]) + '\t' + str(elem[1]) + '\t' + str(elem[2]) + '\t' + str(elem[3]) + '\t' +  str(dnase) + '\t' + str(elem[5]) + '\n'
            i += 1
            if i == len(inpl):
                break
        else:
            break
    if i == len(inpl):
        break
    cnt += 1

for j in range(i, len(inpl)):
    elem = inpl[j]
    line += str(elem[0]) + '\t' + str(elem[1]) + '\t' + str(elem[2]) + '\t' + str(elem[3]) + '\t' + '0' + '\t' + str(elem[5]) + '\n'

out.write(line)

print('Length of ' + sys.argv[3] + ': ' + str(line.count('\n')))

inp.close()
bed.close()
out.close()
