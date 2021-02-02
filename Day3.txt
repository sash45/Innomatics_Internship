Polar Coordinates

import cmath
n=complex(input())
z=complex(n)
print(cmath.polar(z)[0])
print(cmath.polar(z)[1])

Find Angle MBC

import math
AB=int(input())
BC=int(input())
print(str(int(round(math.degrees(math.atan2(AB,BC)))))+'°')

Triangle Quest 2

for i in range(1,int(input())+1): #More than 2 lines will result in 0 score. Do not leave a blank line also
    print((10**i//9)**2)
		

Mod Divmod

a=int(input())
b=int(input())
print(a//b)
print(a%b)
print(divmod(a,b))


Power - Mod Power

import math

a=int(input())
b=int(input())
m=int(input())
print(pow(a,b))
print(pow(a,b,m))

Integers Come In All Sizes

a,b,c,d=(int(input()) for _ in range(4))
print(pow(a,b)+pow(c,d))

Triangle Quest

for i in range(1,int(input())): #More than 2 lines will result in 0 score. Do not leave a blank line also
    print((10**(i)//9)*i)