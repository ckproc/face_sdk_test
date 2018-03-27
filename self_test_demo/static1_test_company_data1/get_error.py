import os,glob,sys

with open(r'testdata.txt') as f:
 s=f.read()
s=[x.strip().split(" ") for x in s.strip().split("\n")]

with open('error.txt', 'w+') as f:
  for i in range(len(s)):
    score=float(s[i][0])
    match=s[i][1]
    a=s[i][2]
    b=s[i][3]
    if match =="1":
      if(score<0.6):
       filepath=s[i][0]+" "+a+" "+b+ '\n'
       f.write(filepath)
    else:
      if(score>=0.4):
       filepath=s[i][0]+" "+a+" "+b+ '\n'
       #f.write(filepath)


