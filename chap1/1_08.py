# 与えられた文字列の各文字を
# 英小文字ならば(219 - 文字コード)の文字に置換
# その他の文字はそのまま出力する関数

def cipher(s):
    s_list = []
    ans = ''

    for i in range(len(s)):
        if s[i].islower():
            s_list.append(chr(219 - ord(s[i])))
        else:
            s_list.append(s[i])

    for n in range(len(s)):
        ans += s_list[n]
        
    return ans

print(cipher('ABCabc'))
print(cipher('zyx'))