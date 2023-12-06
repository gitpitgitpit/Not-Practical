# -*- coding: utf-8 -*-
"""matplot

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1-ZTQ2WUzjgCumrdNUO20-CeEho6jFIPI
"""

import numpy as np
import matplotlib.pyplot as plt
xpoint=np.array([0,6])
ypoint=np.array([0,255])
plt.plot(xpoint,ypoint)
plt.show()

plt.plot(xpoint,ypoint,'o')

xpoint=np.array([50,40,20,70])
ypoint=np.array([23,45,13,45])
plt.plot(xpoint,ypoint)
plt.show()

x=np.array([1,2,3,4,5,6,7,8,9])
y=np.array([10,20,30,40,50,60,70,80,90])
plt.plot(x,y)
plt.title("Travelling")
plt.xlabel("distance")
plt.ylabel("speed")
plt.show()

x=np.array([1,2,3,4,5,6,7,8,9])
y=np.array([10,20,30,40,50,60,70,80,90])
plt.plot(x,y)
plt.title("Travelling")
plt.xlabel("distance")
plt.ylabel("speed")
plt.grid()
plt.show()

x=np.array([1,2,3,4,5,6,7,8,9])
y=np.array([10,20,30,40,50,60,70,80,90])
plt.plot(x,y,linestyle="dotted")
plt.title("Travelling")
plt.xlabel("distance")
plt.ylabel("speed")
plt.show()

x=np.array([1,2,3,4,5,6,7,8,9])
y=np.array([10,20,30,40,50,60,70,80,90])
plt.plot(x,y,linestyle="dashed")
plt.title("Travelling")
plt.xlabel("distance")
plt.ylabel("speed")
plt.show()

x=np.array([1,2,3,4,5,6,7,8,9])
y=np.array([10,20,30,40,50,60,70,80,90])
plt.bar(x,y)
plt.title("Travelling")
plt.xlabel("distance")
plt.ylabel("speed")
plt.show()

x=np.array([1,2,3,4,5,6,7,8,9])
y=np.array([10,20,30,40,50,60,70,80,90])
plt.scatter(x,y)
plt.title("Travelling")
plt.xlabel("distance")
plt.ylabel("speed")
plt.show()

x=np.random.normal(170,10,250)
plt.hist(x)
plt.show()

y=np.array([10,30,50,10])
plt.pie(y)
plt.show()

mylabel=['A','B','C','D']
plt.pie(y,labels= mylabel)
plt.show()

mylabel=['A','B','C','D']
plt.pie(y,labels= mylabel,startangle=90)
plt.show()

mylabel=['A','B','C','D']
plt.pie(y,labels= mylabel,startangle=90,shadow=True)
plt.show()

mylabel=['A','B','C','D']
mycolors=["red","black","pink","grey"]
plt.pie(y,labels= mylabel,startangle=90,shadow=True,colors=mycolors)
plt.show()