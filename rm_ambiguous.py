f = open(sys.argv[1])

l = f.readlines()

for i in range(len(l)):   
    l[i] = l[i].split(',')

final = []

for i in range(len(l)):                                                         
    if l[i][3] == 'yes\n':          
        if (l[i][0] == l[i-1][0]) and (l[i][1] == l[i-1][1]) and (l[i][2] == l[i-1][2]):
            continue
        else:
            final.append(l[i][0]+','+l[i][1]+','+l[i][2]+','+l[i][3])
    else:
        final.append(l[i][0]+','+l[i][1]+','+l[i][2]+','+l[i][3])

w = open(sys.argv[1] + '_final.csv', 'w')

res = ''

for el in final:
    res += el   
w.write(res)

w.close()
f.close()
