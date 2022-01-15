#Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

#datas
x=np.array([0,1,3.4,7]) 
y=np.array([0,5,10.023,20.037])

X=x.reshape(-1,1)
Y=y.reshape(-1,1)

reg=LinearRegression()
reg.fit(X,Y)

#prints interception on x-axis
print(reg.intercept_)
#prints coefficient of the formula 
print(reg.coef_)
#to print 4 significant digits after comma, we used {:.4}
print("Formula of the linear model is: Y = {:.4} + {:.4}X".format(reg.intercept_[0], reg.coef_[0][0])) 

predicted = reg.predict(X)
#Arranging Figure Size
plt.figure(figsize=(12, 6))
#Putting Dots
plt.scatter(x,y,c='orange')
#Drawing Line
plt.plot(X,predicted,c='green', linewidth=3)
#Naming Axes
plt.ylabel("Velocity(V)")
plt.xlabel("Time(t)")
plt.title("Experiment Name")

plt.show()
plt.savefig(fname="exp_1.png",facecolor="#f0f9e8",dpi=600,quality=95)