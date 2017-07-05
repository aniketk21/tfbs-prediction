'''
    mod_peak.py
    usage: python mod_peak.py narrowPeak_file dataset.dat output.dat
    modify the `chipseq_peak` values to 1 if a window from narrowPeak file appears in the dataset.
'''

import sys

nps = open(sys.argv[1])
inp = open(sys.argv[2])
out = open(sys.argv[3], 'w')

inpl = inp.readlines()
npsl = nps.readlines()

for i in range(len(inpl)):
    inpl[i] = inpl[i].split()

for i in range(len(npsl)):
    npsl[i] = npsl[i].split()

#del npsl[0] # this element has all the column names, not actual values

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
    # this is what we've to search for in the inpl_file

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

# extract first column from inpl
first_col_in_inpl = [int(el[0]) for el in inpl]
peak_cnt = 0
for el in npsl:
    # scale num
    num = int(el[1])
    num_inpl = scale(num)
    if num_inpl == 0:
        num_inpl = num

    # now search for num_inpl in the first_col_in_inpl array
    index = bin_search(num_inpl, first_col_in_inpl)
    if index != -1:
        if el[3] == 'B': # if bound.
            inpl[index][6] = '1'
            peak_cnt += 1
        '''
        prev = inpl[index][5]
        if float(el[6]) > float(prev):
            inpl[index][-1] = el[6]
            peak_cnt += 1
        '''
    else:
        print "Not found", num_inpl

for el in inpl:
    res += el[0] + '\t' + el[1] + '\t' + el[2] + '\t' + el[3] + '\t' + el[4] + '\t' + el[5] + '\t' + el[6] + '\n'

print('Number of peaks: ' + str(peak_cnt))

out.write(res)

out.close()
nps.close()
inp.close()
