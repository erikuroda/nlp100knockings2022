# 与えられたシーケンス（文字列やリストなど）からn-gramを作る関数
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

        
s = 'I am an NLPer'

print('------------\n単語bi-gram（リスト）\n------------')
print(make_n_gram(s, 2, 0))

print('\n------------\n文字bi-gram（集合）\n------------')
print(make_n_gram(s, 2, 1))