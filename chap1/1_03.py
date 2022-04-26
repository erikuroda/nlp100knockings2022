# 以下の文を単語に分解し、各単語の文字数を先頭から出現順に並べたリストを作成する
s = 'Now I need a drink, alcoholic of course, after the heavy lectures involving quantum mechanics.'
s = s.replace('.', '')
s = s.replace(',', '')

list1 = s.split(' ')
list2 = []
for elem in list1:
    list2.append(len(elem))

print(list2)