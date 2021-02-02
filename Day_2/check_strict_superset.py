a = set(input().split())
count , n = 0, int(input())
for i in range (n):
        b = set(input().split())
        if a.issuperset(b) :
                count += 1
print(count == n)
