'''
    mod_motif.py
    usage: python mod_motif.py dataset.dat gff_file output.dat
    this will modify the motif scores, found in the `gff_file`
'''

import sys

mchr = open(sys.argv[1])
gff = open(sys.argv[2])
out = open(sys.argv[3], 'w')

mchrl = mchr.readlines()
gffl = gff.readlines()

for i in range(len(mchrl)):
    mchrl[i] = mchrl[i].split()

for i in range(len(gffl)):
    gffl[i] = gffl[i].split()

wset = []
m = []

res = ''
def scale(num):
    new_num = 0
    last_two = num % 100
    if last_two != 0:
        if last_two <= 50:
            if last_two >= 25:
                new_num = num + 50 - last_two # scale up to 50
            else:
                new_num = num - last_two # scale down to 00
        else:
            if last_two >= 75:
                new_num = num + 100 - last_two # scale up to 00
            else:
                new_num = num - (last_two % 50) # scale down to 50
    return new_num
    # this is what we've to search for in the mchrl_file

def bin_search(num, data):
    low = 0
    high = len(data) - 1
    
    while low <= high:
        mid = low + (high - low) / 2

        if data[mid] == num:
            return mid
        elif data[mid] < num:
            low = mid + 1
        else:
            high = mid - 1

    return -1

# extract first column from mchrl
first_col_in_mchrl = [int(el[0]) for el in mchrl]
motif_cnt = 0
for el in gffl:
    # scale num
    num = int(el[3])
    num_mchrl = scale(num)
    if num_mchrl == 0:
        num_mchrl = num

    # now search for num_mchrl in the first_col_in_mchrl array
    index = bin_search(num_mchrl, first_col_in_mchrl)
    if index != -1:
        # motif present or not?
        #if ('myc' in el[-1]) or ('MYC' in el[-1]):
        if 'E2F' in el[-1]:    
            #mchrl[index][4] = '1'
            motif_cnt += 1
            
            # extract prev score
            prev = mchrl[index][3]
            if float(el[5]) > float(prev):
                mchrl[index][3] = el[5]
    else:
        print "Not found", num_mchrl

for el in mchrl:
    res += el[0] + '\t' + el[1] + '\t' + el[2] + '\t' + el[3] + '\t' + el[4] + '\t' + el[5] + '\n'

print('Number of motifs = ' + str(motif_cnt))

out.write(res)

out.close()
gff.close()
mchr.close()
