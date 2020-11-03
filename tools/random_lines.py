#!/usr/bin/env  python
import random, os, sys
from sets import Set
f = open(sys.argv[1])
w = open(sys.argv[2], 'w')

sample_num = int(sys.argv[3])
query_num  = int(os.popen('wc -l %s'%sys.argv[1]).read().split()[0])
print query_num
l = range(query_num)
u = random.sample(l, sample_num)
u = Set(u)

num = 0
col = -1

if (len(sys.argv) > 4):
    col = int(sys.argv[4])
    
for line in f:
    if (num in u):
        if (col == -1):
            w.write("%s"%line)
        else:
            l = line.strip().split()[col]
            w.write("%s\n"%l)
    num += 1
    if (num % 1000 == 0):
        print num
                
