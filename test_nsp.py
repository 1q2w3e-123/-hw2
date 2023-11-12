import numpy as np

# print(np.__version__)
l=[1,2,3,4]
l1=[1,2,3]
l=np.array(l)
l1=np.array(l1)
x=[]
x.append(l)
x.append(l1)
x=np.array(x)
y=[]
y.append(x)
y.append(x)
print(y)