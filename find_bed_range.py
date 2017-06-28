'''
    find_bed_range.py
    usage: python find_bed_range.py bed_file.bed
    find `low1`, `low2` and `high2` to be given to create_bp_windows.py
'''

import sys

f = open(sys.argv[1])

l = f.readlines()

for i in range(len(l)):
    l[i] = l[i].split()
    l[i][1] = int(l[i][1])
    l[i][2] = int(l[i][2])

min = sys.maxint
max = -min -1
max_idx = 0

for i in range(len(l)):
    if l[i][1] < min:
        min = l[i][1]
    if l[i][2] > max:
        max = l[i][2]
        max_idx = i

print('low1: ' + str(min))
print('low2: ' + str(l[max_idx][1]))
print('high2: ' + str(max))

f.close()
