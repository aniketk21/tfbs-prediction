'''
    create_bp_windows.py
    usage: python create_bp_windows.py output_file.dat low1 low2 high2
    create 200 base pair windows from `low1` to `low2`, saved in `output_file.dat`
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
        chrStart    chrEnd  chrom_state motif_score motif   chipseq_peak
    '''
    f.write(str(low1) + '\t' + str(high1) + '\t' + '0' + '\t' + '0' + '\t' + '0' + '\t' + '0')
    f.write('\n')
    low1 += 50
    high1 += 50
    cnt += 1

print('Length of ' + sys.argv[1] + ': ' + str(cnt))

f.close()
