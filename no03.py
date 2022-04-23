list1="Now I need a drink, alcoholic of course, after the heavy lectures involving quantum mechanics."
list2=list1

list2=list2.replace(',',"")
list2=list2.replace('.'," .")
list2=list2.split()
list=[]
for i in range(len(list2)):
    list.append(len(list2[i]))
print(list)
