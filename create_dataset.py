'''
    create_data.py
    create 200 base pair intervals from `low1` to `low2`, saved in `filename.dat`
    usage: python create_data.py filename.dat low1 low2 high2
    round `low1` to the nearest multiple of 50
'''

import sys

f = open(sys.argv[1], 'w')

# Bounds for dataset
low1 = int(sys.argv[2])
high1 = low1 + 200
low2 = int(sys.argv[3])
high2 = int(sys.argv[4])

cnt = 0
while low1 <= low2 and high1 <= high2:
    '''
        chrStart    chrEnd  chrom_state motif_score chipseq_score   bind
    '''
    f.write(str(low1) + '\t' + str(high1) + '\t' + '0' + '\t' + '0' + '\t' + '0' + '\t' + '0')
    f.write('\n')
    low1 += 50
    high1 += 50
    cnt += 1

print('Number of lines in ', sys.argv[1], '= ', cnt)

f.close()
