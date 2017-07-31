f = open('data_list.txt', 'w')

with open('neg_48.txt', 'r') as f1:
    a1 = f1.readlines()
idx = 0
count1 = 0
for a in a1:
    if (idx % 1000 == 0):
        print idx, "neg done"
        print count1, "real neg done"
    if (idx % 3 == 0):
        f.write("%s" % a)
        count1 = count1 + 1
    idx = idx + 1

with open('pos_48.txt', 'r') as f1:
    a1 = f1.readlines()
idx = 0
count2 = 0
for a in a1:
    if (idx % 1000 == 0):
        print idx, "pos done"
        print count2, "real pos done"
    if (idx % 14 == 0):
        f.write("%s" % a)
        count2 = count2 + 1
    idx = idx + 1

with open('part_48.txt', 'r') as f1:
    a1 = f1.readlines()
idx = 0
count3 = 0
for a in a1:
    if (idx % 1000 == 0):
        print idx, "part done"
        print count3, "real part done"
    if (idx % 38 == 0):
        f.write("%s" % a)
        count3 = count3 + 1
    idx = idx + 1

with open('celeba_48.txt', 'r') as f1:
    a1 = f1.readlines()
idx = 0
count4 = 0
for a in a1:
    f.write("%s" % a)
    count4 = count4 + 1

f.close()

print "whole data"
print count1, "neg"
print count2, "pos"
print count3, "part"
print count4, "celeba"
