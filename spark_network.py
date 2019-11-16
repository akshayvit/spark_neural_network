from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('akshay').getOrCreate()
df=spark.read.csv('sadw\\actors_emotions.csv',header=True)
headers=df.schema.names
features=[]
target=[]
length=len(df.select("zcr").collect())
for i in range(length):
    list=[]
    for j in range(len(headers)-1):
        list.append(float(df.select(headers[j]).collect()[i][headers[j]]))
    features.append(list)
    target.append([int(df.select(headers[len(headers)-1]).collect()[i].tar)])
import sys
from random import *
import numpy as np
from math import *
import time
from matplotlib import pyplot as plt
class  Neural_Network(object):
    def __init__(self):
        self.inputlayers=7
        self.outputlayers=1
        self.hiddenlayers=8
        self.w1=np.random.randn(self.inputlayers,self.hiddenlayers)
        self.w2=np.random.randn(self.hiddenlayers,self.outputlayers)
    def forward(self,X):
        self.z2=np.dot(X,self.w1)
        self.a2=self.sigmoid(self.z2)
        self.z3=np.dot(self.a2,self.w2)
        yvect=self.sigmoid(self.z3)
        return yvect
    def sigmoid(self,Z):
        return 1/(1+np.exp(-Z))
X=np.array(features[:length-1])
nn=Neural_Network()
yvect=nn.forward(X)
print("\nThe final precised vector is:\t"+str(yvect))
weights=np.linspace(-10,10,1000)
costs=np.zeros(1000)
starttime=time.clock()
y=np.array(target[:length-1])
print("Our actul assumption is:\t"+str(y))
for i in range(1000):
    nn.w1[0,0]=weights[i]
    yvect=nn.forward(X)
    costs[i]=0.5*sum((y-yvect)**2)
print(costs)
endtime=time.clock()
timeelapsed=endtime-starttime
print(timeelapsed)
x=np.array(features[length-1])
print("\n The minimum cost is:\t"+str(min(costs)))
plt.plot(weights,costs,color="g")
plt.show()
