import random

list="I couldn’t believe that I could actually understand what I was reading : the phenomenal power of the human mind."

list=list.replace("."," . ")
list=list.split()
#print(len(list))
new_list=[]
for i in range(len(list)):
    if len(list[i])<5:
        new_list.append(list[i])
    else:
        word_list=[]#並び替えた単語
        word_list.append(list[i][0])
        word=[]#ランダムな文字列
        #2からlen(list[])-1までをランダムに並び替える
        for j in range(len(list[i])-2):
            word.append(list[i][j+1])
        word=random.sample(word,len(word))
            
        for k in range(len(word)):
            word_list.append(word[k])
        word_list.append(list[i][len(list[i])-1])
        word_list=''.join(word_list)
        new_list.append(word_list)
new_list=' '.join(new_list)
print(new_list)
        
        
            
            
        

