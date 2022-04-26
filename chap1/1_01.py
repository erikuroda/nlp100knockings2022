# 文字列の奇数番目を取り出す
text1 = 'パタトクカシーー'
list1 = list(text1)
list2 = []

for i in range(len(list1)):
    if i%2==0:
        list2.append(list1[i])

text2 = ''.join(list2)
print(text2)