def make_n_gram(text, n, type):
    text = text.replace(',', '')
    text = text.replace('.', '')

    if type==0:
        org_list = text.split(' ')
        n_word_list = []
        # n_word_set = set()
        for i in range(len(org_list)-(n-1)):
            n_word_list.append(org_list[i:i+n])
            # n_word_set.add(str(org_list[i:i+n]))
        return n_word_list
        # return n_word_set
    
    elif type==1:
        text = text.replace(' ', '')
        # n_ch_list = []
        n_ch_set = set()

        for i in range(len(text)-(n-1)):
            # n_ch_list.append(text[i:i+n])
            n_ch_set.add(text[i:i+n])
        # return n_ch_list
        return n_ch_set


str1 = 'paraparaparadise'
str2 = 'paragraph'
X = make_n_gram(str1, 2, 1)
Y = make_n_gram(str2, 2, 1)

# 和集合
U = X | Y
print(U)

# 積集合
I = X & Y
print(I)

# 差集合
D = X - Y
print(D)

print('se' in X)
print('se' in Y)