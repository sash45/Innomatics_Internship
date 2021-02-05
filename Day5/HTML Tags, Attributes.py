from html.parser import HTMLParser
class Parser(HTMLParser):
    def handle_starttag(self, tag, attrs):
        print(tag)
        for attr in attrs:
            prop,value = attr
            print("-> " + prop + " > " + value)
N=int(input())
s=""
for i in range(N):
    s+= " " + input()
parser = Parser()
parser.feed(s)
