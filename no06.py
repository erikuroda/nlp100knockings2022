X="paraparaparadise"
Y="paragraph"

n=2
bi_X=[]
bi_Y=[]
        

for i in range(len(X)-n+1):
    list=[]
    for j in range(n):        
        list.append(X[i+j])
    if list not in bi_X:
        bi_X.append(list)

for i in range(len(Y)-n+1):
    list=[]
    for j in range(n):        
        list.append(Y[i+j])
    if list not in bi_Y:
        bi_Y.append(list)

print('bi_X\n',bi_X)
print('bi_Y\n',bi_Y)


wa=[]
for bi in bi_Y:
    wa.append(bi)
for bi in bi_X:
    if bi not in bi_Y:
        wa.append(bi)
      
print('wa\n',wa)


seki=[]
for X in bi_X:
    for Y in bi_Y:
        seki_list1=[]
        seki_list2=[]
        seki_list1.append(X)
        seki_list2.append(Y)
        seki_list1.append(Y)
        seki_list2.append(X)
        if seki_list1 not in seki and seki_list2 not in seki:
            seki.append(seki_list1)
        

print('seki\n',seki)

sa=[]
for X in bi_X:
    if X not in bi_Y:
        sa.append(X)

for Y in bi_Y:
    if Y not in bi_X:
        sa.append(Y)
        
print('sa\n',sa)

