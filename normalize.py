from sklearn.preprocessing import  Normalizer

x = [[1,2,3],[3,2,1],[3,6,8]]

trans = Normalizer(norm = 'l2').fit(x)#fit does nothing 

data  = trans.transform(x)
print(data)