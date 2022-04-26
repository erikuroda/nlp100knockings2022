# 以下の文sを単語に分解し，
# 1, 5, 6, 7, 8, 9, 15, 16, 19番目の単語は先頭の1文字，
# それ以外の単語は先頭の2文字を取り出し，
# 取り出した文字列から単語の位置（先頭から何番目の単語か）への連想配列（辞書型もしくはマップ型）を作成する．

# start以降を削除する関数
def remove_str_start(s, start):
    return s[:start]

s = 'Hi He Lied Because Boron Could Not Oxidize Fluorine. New Nations Might Also Sign Peace Security Clause. Arthur King Can.'
s = s.replace('.', '')
list1 = s.split(' ')
# print(list1)

list2 = []
for i in range(len(list1)):
    n = i+1
    if n==1 or n==5 or n==6 or n==7 or n==8 or n==9 or n==15 or n==16 or n==19:
        list2.append(remove_str_start(list1[i], 1))
    else:
        list2.append(remove_str_start(list1[i], 2))

# print(list2)

# 連想配列作成
d = {}
for i in range(len(list2)):
    d[list2[i]] = i+1

print(d)
