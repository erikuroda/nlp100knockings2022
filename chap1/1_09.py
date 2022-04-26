# Typoglycemia
# 長さが4以下の単語に対しては何もしない
# 長さが5以上の単語に対して、先頭末尾以外をランダムに並べ替える

import random

def Typoglycemia(s):
    word_list = s.split(' ')
    ans = ''

    for i in range(len(word_list)):
        w = word_list[i]
        if len(w) > 4:
            first = w[0]
            end = w[-1]
            mid = w[1:-1]
            # 先頭末尾以外をランダムに並べ替えた文字列作成
            midrd = ''.join(random.sample(mid, len(mid)))
            ans = first + midrd + end
            word_list[i] = ans

    return word_list


newsr = Typoglycemia('I couldn’t believe that I could actually understand what I was reading : the phenomenal power of the human mind.')
print(newsr)