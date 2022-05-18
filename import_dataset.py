# 50 TODO 解压已下载的zip压缩包，并阅读readme.txt；
import pandas as pd
# /usr/bin/python3 -m pip install pandas
df = pd.read_csv("dataset/newsCorpora.csv")
print(df)
#TODO 提取由 “Reuters”，”Huffington Post”，”Businessweek”，”Contactmusic.com” 及”Daily Mail”所出版的文章语料；
#TODO 随机打乱所提取出的实例（即文章）的顺序；
#TODO 以训练集占80%、验证集占10%、测试集占10%的比例分割所提取出的语料，然后分别存储至文件train.txt，valid.txt和test.txt。
#TODO 文件中，每行记录一个实例，每个实例记录该实例所属的类别与文章的标题，二者之间以制表符（TAB）分隔。source .venv/bin/activate