# 二つの文字列から文字を交互に連結する
s1 = 'パトカー'
s2 = 'タクシー'

l1 = list(s1)
l2 = list(s2)

anslst = []

for i in range(len(s1)):
    anslst.append(l1[i])
    anslst.append(l2[i])

anstxt = ''.join(anslst)
print(anstxt)