list="I am an NLPer."

def cipher(list):
    list=list.replace("."," .")
    list=list.split(" ")
    list1=[]    
    for i in range(len(list)):
        word_list=[]
        for j in range(len(list[i])):
            if list[i][j].islower():
                word_list.append(chr(219-ord(list[i][j])))
            else:
                word_list.append(list[i][j])
        word_list=''.join(word_list)
        list1.append(word_list)
    list1=' '.join(list1)

    return list1

print(cipher(list))

    
