import pandas as pd

df = pd.read_table('./popular-names.txt', header=None, sep='\t', names=['name', 'sex', 'number', 'year'])
print(len(df))

#2780

# 確認
#!wc -l ./popular-names.txt
#2780 ./popular-names.txt
