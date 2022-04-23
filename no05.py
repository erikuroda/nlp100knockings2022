sentence="I am an NLPer"

n=2

sentence1=sentence.split()
list_tango=[]
for i in range(len(sentence1)-n+1):
    list1=[]
    for j in range(n):        
        list1.append(sentence1[i+j])
    if list1 not in list_tango:
        list_tango.append(list1)

print('単語bi-gram',list_tango)


sentence2=sentence.replace(" ","")
list_moji=[]
for i in range(len(sentence2)-n+1):
    list2=[]
    for j in range(n):        
        list2.append(sentence2[i+j])
    if list2 not in list_moji:
        list_moji.append(list2)

print('文字bi-gram',list_moji)
