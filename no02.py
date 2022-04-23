list1="パトカー"
list2="タクシー"

list=[]
for i in range(len(list1)):
    list.append(list1[i])
    list.append(list2[i])
new_list=''.join(list)
print(new_list)
