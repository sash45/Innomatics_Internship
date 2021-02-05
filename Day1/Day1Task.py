Program 1:

print("Hello, World!")

Program2:

import math
import os
import random
import re
import sys


if __name__ == '__main__':
    n = int(input().strip())
    if n % 2 != 0:
        print("Weird")
    else:
        if n >= 2 and n <= 5:
            print("Not Weird")
        elif n >= 6 and n <= 20:
            print("Weird")
        elif n > 20:
            print("Not Weird")
            
Program 3:
if __name__ == '__main__':
    a = int(input())
    b = int(input())
    print(a+b)
    print(a-b)
    print(a*b)

Program4:
if __name__ == '__main__':
    a = int(input())
    b = int(input())
    print(round(a/b))
    print(a/b)

Program5:
if __name__ == '__main__':
    n = int(input())
    for i in range(n):
        print(i*i)

Program6:
def is_leap(year):
    if year % 400 == 0:
        return True
    if year % 100 == 0:
        return False
    if year % 4 == 0:
        return True
    else:
        return False

year = int(input())
print(is_leap(year))

Program7:
def is_leap(year):
    if year % 400 == 0:
        return True
    if year % 100 == 0:
        return False
    if year % 4 == 0:
        return True
    else:
        return False

year = int(input())
print(is_leap(year))

