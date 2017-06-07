f = open('gm12878_chr22.dat', 'w')

# Bounds for dataset
low1 = 16050000
high1 = 16050200
low2 = 51234702
high2 = 51244566

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

cnt = 0
while low1 <= low2 and high1 <= high2:
    f.write(str(low1) + '\t' + str(high1) + '\t' + '0' + '\t' + '0' + '\t' + '0' + '\t' + '0')
    f.write('\n')
    low1 += 50
    high1 += 50
    cnt += 1

print(cnt)
f.close()
