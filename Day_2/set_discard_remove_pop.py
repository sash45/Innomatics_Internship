n = int(input())
s = set(map(int, input().split()))
N=int(input())
for i in range(N) :
    comm=input().split()
    if comm[0]=="remove" :
        s.remove(int(comm[1]))
    elif comm[0]=="discard" :
        s.discard(int(comm[1]))
    else:
        s.pop()
print (sum(list(s)))
