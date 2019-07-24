from __future__ import division
from pylab import *
from scipy.optimize import fsolve



def potential(x, n, c2, c4):
    result = []
    for i in range(n):
        i1 = range(i)
        i2 = arange(n-i-1) + i + 1
#        print 1/(x[i]-x[i1])**2
#        print sum(1/(x[i]-x[i2])**2)
        u = 2 * x[i]**3 * c4 + x[i] * c2 - sum(1/(x[i]-x[i1])**2) + sum(1/(x[i]-x[i2])**2)
        result.append(u)
    return array(result)



N = 30
Nc= 50
C_l = linspace(0,5.1,Nc)

x0 = arange(N) - N/2.0

c2 = 1
c4 = C_l[3]
a = fsolve(potential,x0,args=(N, c2, c4))
x = a

dlist = []

for ci in range(Nc):
    c4= C_l[ci]
    a = fsolve(potential,x0,args=(N, c2, c4),xtol=1e-13)
    x0 = a
    print var(a)
    #plot(a/max(a),zeros((N,1))+c4,'ro')
    dlist.append(a/max(a))

dlist= array(dlist)
for i in range(N):
    plot(dlist[1:,i],C_l[1:],'r-x')
show()
