T= int(input())
for i in range(T):
    n=int(input())
    A=set(input().split())
    m=int(input())
    B=set(input().split())
    if A.issubset(B):
        print(True)
    else:
        print(False)
