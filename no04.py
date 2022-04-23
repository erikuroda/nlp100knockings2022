list1="Hi He Lied Because Boron Could Not Oxidize Fluorine. New Nations Might Also Sign Peace Security Clause. Arthur King Can."
list2=list1

list2=list2.replace('.',"")
list2=list2.split()

dictionary={}
for i in range(len(list2)):
    if i==0 or (i>3 and i<9) or i==14 or i==15 or i==18:
        dictionary[list2[i]]=list2[i][0]
        
    else:
        dictionary[list2[i]]=''.join([list2[i][0],list2[i][1]])
        

print(dictionary)

