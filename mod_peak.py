inp = open('gm12878_chr22_mod_motif_score.dat')
nps = open('GM12878_MYC.narrowPeak')
out = open('gm12878_chr22_mod_npscore.dat', 'w')

i = 0

inpl = inp.readlines()
npsl = nps.readlines()

for _ in range(len(inpl)):
    inpl[_] = inpl[_].split()

for _ in range(len(npsl)):
    npsl[_] = npsl[_].split()

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
cnt = 0
for el in npsl:
    #remove_from_inpl(res, el[1], i)
    # scale num
    num = int(el[1])
    num_inpl = scale(num)
    if num_inpl == 0:
        num_inpl = num
    #print(num_inpl)

    # now search for num_inpl in the first_col_in_inpl array
    index = bin_search(num_inpl, first_col_in_inpl)
    if index != -1:
        # extract prev score
        inpl[index][-1] = '1'
        cnt += 1
        prev = inpl[index][4]
        if float(el[4]) > float(prev):
            inpl[index][4] = el[4]
#TEST CASE        if num_inpl == 22336900:
#            print 'prev', prev, 'inpl', inpl[index][3]
    else:
        print "Not found", num_inpl

for el in inpl:
    res += str(el) + '\n'
print(cnt)
out.write(res)
out.close()
nps.close()
inp.close()
