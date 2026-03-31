import numpy as np
print("MATRIX",end = "\n\n")

l = [[1,2,3],[5,6,7],[0,0,0]]
m = np.array(l)
n = np.array([[9,1,2],[3,4,5],[7,5,3]])
print(m)

print()

print("First row is {}".format(m[0]))
print("Third column is {} ".format(m[:,2]))

print()
print()
print()
print("m\n{}".format(m))
print("n\n{}".format(n))

print("n+m\n{}".format(m+n))
print("n*m\n{}".format(m*n))
print("max(n) = {}".format(n.max()))
print("mean(n) = {}".format(n.mean()))
n = n.T
print("n trasposed\n {}".format(n))


x = np.linspace(5,10,100)#prende 100 punti di una retta che vanno da 5 a 10 
for i in x:
    print(i)
