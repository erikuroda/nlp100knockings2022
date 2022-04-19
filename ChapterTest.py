# 00
one_str = 'stressed'
ans = one_str[::-1]
print(ans)

#01
two_str = 'パタトクカシーー'
print(two_str[::2])

#02
word_p = "パトカー"
word_t = "タクシー"

result = ""
for p, t in zip(word_p, word_t):
    result += p + t
print(result)

