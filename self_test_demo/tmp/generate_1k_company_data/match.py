import glob 
import os


querylist=glob.glob('./query/*.jpg')
dblist=glob.glob('./db/*.jpg')
db=[]
query=[]

for file in dblist:
 name=os.path.split(file)[1]
 db.append(name)

for file in querylist:
 name=os.path.split(file)[1]
 query.append(name)
 
i=0
with open('matchlist.txt','w+') as f:
 for name in query:
  if name in db:
   path=name+" "+"1"+"\n"
   i=i+1
  else:
   path=name+" "+"0"+"\n"
  f.write(path)
  
print i