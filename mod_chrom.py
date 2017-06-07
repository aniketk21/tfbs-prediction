g = open('GM12878_CHR22.bed')
f = open('gm12878_chr22.dat')
h = open('gm12878_chr22_mod_chrom.dat', 'w')

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

print(len(f_lines))
print(len(g_lines))

cnt = 0
i = 0
for el in g_lines:
    low = int(el[1])
    high = int(el[2])
    chrom = str(chrom_states[el[3]])
    while True:
        elem = f_lines[i]
        #if int(elem[0]) <= high and int(elem[1]) <= high:
        if int(elem[1]) <= high:
            line = elem[0] + '\t' + elem[1] + '\t' + chrom + '\t' + elem[3] + '\t' +  elem[4] + '\t' + elem[5]
            h.write(line)
            h.write('\n')
            i += 1
            if i == len(f_lines):
                break
        #elif int(elem[1]) >= high:
        #    i += 1
        #    break
        else:
            break
    if i == len(f_lines):
        break
    cnt += 1

print(cnt)
f.close()
g.close()
h.close()
