mchr = open('gm12878_chr22_mod_chrom.dat')
gff = open('homo_sapiens.hg19.chr22.gff')
out = open('gm12878_chr22_mod_motif_score.dat', 'w')
i = 0

mchrl = mchr.readlines()
gffl = gff.readlines()

for _ in range(len(mchrl)):
    mchrl[_] = mchrl[_].split()

for _ in range(len(gffl)):
    gffl[_] = gffl[_].split()

wset = []
m = []

res = ''
'''
def remove_from_mchrl(res, elem, i):
    '''
        #remove all entries from mchrl that have low < elem[3] and append them to the result
    '''
    el = int(elem[3])
    for j in range(i, len(mchrl)):
        elm = int(mchrl[j][0])
        if elm < el:
            #res += str(mchrl[j]) + '\n'
            i += 1
        else:
            return
'''
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
cntr = 0
for el in gffl:
    #check for occurrence of 'myc' in the gff file
    myc_flag = False
    if 'myc' in el[-1] or 'MYC' in el[-1]:
        myc_flag = True
    #remove_from_mchrl(res, el[3], i)
    # scale num
    num = int(el[3])
    num_mchrl = scale(num)
    if num_mchrl == 0:
        num_mchrl = num
    #print(num_mchrl)

    # now search for num_mchrl in the first_col_in_mchrl array
    index = bin_search(num_mchrl, first_col_in_mchrl)
    if index != -1:
        if myc_flag:
            mchrl[index][-1] = '1'
        # extract prev score
        prev = mchrl[index][3]
        if float(el[5]) > float(prev):
            mchrl[index][3] = el[5]
#TEST CASE        if num_mchrl == 22336900:
#            print 'prev', prev, 'mchrl', mchrl[index][3]
    else:
        if myc_flag:
            cntr += 1
        print "Not found", num_mchrl

for el in mchrl:
    res += str(el) + '\n'
out.write(res)
out.close()
gff.close()
mchr.close()
