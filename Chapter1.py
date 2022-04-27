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


#03
sentence = "Now I need a drink, alcoholic of course, after the heavy lectures involving quantum mechanics."
sentence = s.replace(',','').replace('.','')
[len(w) for w in sentence.split()]


#4
s = "Hi He Lied Because Boron Could Not Oxidize Fluorine. New Nations Might Also Sign Peace Security Clause. Arthur King Can."
s = s.replace('.','')
idx = [1, 5, 6, 7, 8, 9, 15, 16, 19]
mp = {}
for i,w in enumerate(s.split()):
    if (i+1) in idx:
        v = w[:1]
    else:
        v = w[:2]
    mp[v] = i+1
print (mp)

#5
def ngram(S, n):
    r = []
    for i in range(len(S) - n + 1):
        r.append(S[i:i+n])
    return r
s = 'I am an NLPer'
print (ngram(s.split(),2))
print (ngram(s,2))

#6
def ngram(S, n):
    r = []
    for i in range(len(S) - n + 1):
        r.append(S[i:i+n])
    return r
s1 = 'paraparaparadise'
s2 = 'paragraph'
st1 = set(ngram(s1, 2))
st2 = set(ngram(s2, 2))
print(st1 | st2)
print(st1 & st2)
print(st1 - st2)
print('se' in st1)
print('se' in st2)

#7
def temperature(x,y,z):
    return str(x)+'時の'+str(y)+'は'+str(z)
x = 12
y = '気温'
z = 22.4
print (temperature(x,y,z))

#8
def cipher(S):
    new = []
    for s in S:
        if 97 <= ord(s) <= 122:
            s = chr(219 - ord(s))
        new.append(s)
    return ''.join(new)

s = 'I am an NLPer'
new = cipher(s)
print (new)

print (cipher(new))

#9
import random
s = 'I couldn’t believe that I could actually understand what I was reading : the phenomenal power of the human mind .'
ans = []
text = s.split()
for word in text:
    if (len(word)>4):
        mid = list(word[1:-1])
        random.shuffle(mid)
        word = word[0] + ''.join(mid) + word[-1]
        ans.append(word)
    else:
        ans.append(word)
print (' '.join(ans))