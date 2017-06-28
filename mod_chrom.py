'''
    mod_chrom.py
    usage: python mod_chrom.py bed_file.bed dataset.dat output.dat
    modify the `chrom_state` values changing the 0 to one of (1, 2, 3, 4, 5, 6, 7)
'''

import sys

g = open(sys.argv[1])
f = open(sys.argv[2])
h = open(sys.argv[3], 'w')

f_lines = f.readlines()
g_lines = g.readlines()

# dictionary of chromatin states
chrom_states = {
        'TSS'   : 1,
        'PF'    : 2,
        'E'     : 3,
        'WE'    : 4,
        'CTCF'  : 5,
        'T'     : 6,
        'R'     : 7
}

for i in xrange(len(g_lines)):
    g_lines[i] = g_lines[i].split()

for i in xrange(len(f_lines)):
    f_lines[i] = f_lines[i].split()

print('Length of ' + sys.argv[1] + ': ' + str(len(g_lines)))
print('Length of ' + sys.argv[2] + ': ' + str(len(f_lines)))

cnt = 0
i = 0
line = ''
for el in g_lines:
    low = int(el[1])
    high = int(el[2])
    chrom = chrom_states[el[3]]
    while True:
        elem = f_lines[i]
        if int(elem[1]) <= high:
            line += str(elem[0]) + '\t' + str(elem[1]) + '\t' + str(chrom) + '\t' + str(elem[3]) + '\t' +  str(elem[4]) + '\t' + str(elem[5]) + '\n'
            i += 1
            if i == len(f_lines):
                break
        else:
            break
    if i == len(f_lines):
        break
    cnt += 1

h.write(line)

print('Length of ' + sys.argv[3] + ': ' + str(line.count('\n')))

f.close()
g.close()
h.close()
