import matplotlib.pyplot as plt
import cv2
#lena = cv2.imread('Jake_Cherry_4.jpg')
#ok=cv2.imread('Jordana_Spiro_3.jpg')
#ok=ok[:,:,::-1]
#lena=lena[:,:,::-1]
#plt.ion() 
with open(r'fpr_tpr.txt') as f:
 s=f.read()
s=[x.strip().split(" ") for x in s.strip().split("\n")]
tpr=[]
fpr=[]
for i in range(len(s)):
 #print float(s[i][0]
 fpr.append(float(s[i][0]))
 tpr.append(float(s[i][1]))
plt.figure(1)  
plt.plot(fpr,tpr)
plt.axis([-0.1,1.1,-0.1,1.1])   
plt.title("roc curve")
#plt.subplot(122)             # the first subplot in the first figure
#plt.imshow(lena)
plt.xlabel('fpr', fontsize=15)
plt.ylabel('tpr', fontsize=15)
plt.savefig("roc.png")
print "the ROC curve have been saved into roc.png"

#plt.plot([1, 2,3])
#plt.subplot(121)             # the second subplot in the first figure
#plt.plot([4, 5, 6])
#plt.figure(2)
#plt.imshow(ok)



plt.show()