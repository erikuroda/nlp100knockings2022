# 引数x, y, zを受け取り「x時のyはz」という文字列を返す関数を実装
# さらに，x=12, y=”気温”, z=22.4として，実行結果を確認

def return_xyz(x, y, z):
    s = f'{x}時の{y}は{z}'
    return s

print(return_xyz(12, '気温', 22.4))