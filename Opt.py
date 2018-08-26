#*************************************************************
'''
import sys;sys.path.append('C:/Draughts/Draughts');import Opt
'''
#*************************************************************
from math import sqrt
import random
import time
import itertools as it
import operator
import range1 as rg
import winsound
#

A=None
NN=None
#*************************************************************
#										Get Date
def GetDate():
	s22=time.asctime().split(' ')
	s33=s22[0]+' '+s22[1]+' '+s22[2]+' '+s22[len(s22)-1]
	return s33.replace(' ','_')
#*************************************************************
#										File2Arr
def File2Arr(f):
	f0=open(f,'r')
	a=[]
	for x in f0:
		x=x[0:len(x)-1]
		a.append(x)
	f0.close()
	return a
#*************************************************************
# length=2
# Max=3
# return [[0,1],[0,2],[1,2]]
def CountN(length,Max):
	return map(list,it.combinations(range(Max),length))
#*************************************************************
# X=[[1,0,0,1],[1,1,0,1]]
# Y=[1,2]
# returns number of occurance that all variable index in Y are true in X
def Count(X,Y):
	k=0
	for x in X:
		t=1
		for y in Y:
			if not x[y]:
				t=0
				break
		k+=t
	return k
#*************************************************************
#										String To Vector
def String2Vector(x):
	a=x.split(' ')
	return [eval(i) for i in a]

def String2Vector_File(f):
	f0=open(f,'r')
	A=[]
	for x in f0:
		x=x.strip()
		A.append(String2Vector(x))
	f0.close()
	return A

def Strings2Data(A_):
	D_=[]
	for x in A_:
		D_.append(String2Vector(x))
	return D_
#*************************************************************
#										Get Time
def GetTime():
	x=time.localtime()
	x22=' am'
	x11=int(x[3])
	x1=''
	if(x11>12):
		x1=str(x[3]-12)+':'
		x22=' pm'
	else:
		if(x[3]==0):
			x1='12:'
		else:
			x1=str(x[3])+':'
	x2=str(x[4])
	x3=':'+str(x[5])
	if(len(x2)==1):
		x2='0'+x2
	return (x1+x2+x3+x22)
#*************************************************************
#										First Partial
def Partial1(x,i,h):
	a=[]
	for j in range(len(x)):
		if(j==i):
			a.append(x[j]+h)
		else:
			a.append(x[j])
	return a
#*************************************************************
#										Second Partial
#*************************************************************
def Partial2(x,i,h):
	a=[]
	for j in range(len(x)):
		if(j==i):
			a.append(x[j]+(2*h))
		else:
			a.append(x[j])
	return a
#*************************************************************
#										First Derivative
#*************************************************************
def Gradient1(f,x,h=0.0001):
	a=[]
	for i in range(len(x)):
		s1=Partial1(x,i,h)
		s2=f(s1)-f(x)
		s2/=h
		a.append(s2)
	return a
#*************************************************************
#										Second Derivative
#*************************************************************
def Gradient2(f,x,h=0.0001):
	a=[]
	for i in range(len(x)):
		s1=Partial1(x,i,h)
		s2=Partial2(x,i,h)
		s11=f(s1)
		s22=f(s2)
		s3=(((s22-s11)/h)-((s11-f(x))/h))/h
		a.append(s3)
	return a
#*************************************************************
#										Norm
#*************************************************************
def Norm(x):
	s=0.0
	for o in x:
		s+=(o*o)
	return sqrt(s)
#*************************************************************
def Distance(x,y):
	s=0.0
	for i in range(len(x)):
		s+=(x[i]-y[i])**2
	return sqrt(s)
#*************************************************************
#										Find Minimum
#*************************************************************
sol=[]
best=9e999
OptimumReached=0
#*************************************************************
def Reset():
	global sol
	global best
	global OptimumReached
	sol=[]
	best=9e999
	OptimumReached=0
#*************************************************************
def Find_Min(f,x,tol=1e-06,h=1e-5,N=99e99):
	global sol
	global best
	n=len(x)
	x0=[i for i in x]
	i=0
	best=9e999
	d=0
	while True:
		i+=1
		g1=Gradient1(f,x0,h)
		eval0=f(x0)
		try:
			x0=[(x0[k]-(eval0/(g1[k]))) for k in range(len(x0))]
		except:
			print 'Error : Gradient Division By 0'
			d=1
			break
		eval1=f(x0)
		if(eval1<best):
			sol=x0
			best=eval1
			print('*Current Best ('+str(best)+')')
		if(eval1<=tol):
			sol=x0
			best=eval1
			break
		if(i>=N):
			break
	if d:
		print('Terminated With Error : Gradient Division By 0 , at Iteration '+str(i)+" With Error "+str(best))
	else:
		print('Terminated Successfully at Iteration '+str(i)+" With Error "+str(best))
	return x0
#*************************************************************
def IncrPoint(Origin_,Direction_,Length_):
	X=[]
	for i in range(len(Origin_)):
		X.append(Origin_[i]+Direction_[i]*Length_)
	return X
#*************************************************************
#										Random Optimization
#*************************************************************
def Normalize(x):
	l=Norm(x)+1e-100
	return [(i/l) for i in x]
#*************************************************************
def Normalize_Prob(x,k=2):
	s=sum(x)+0.0
	return [round(i/s,k) for i in x]
#*************************************************************
def RandomVector(n):
	x=[random.triangular(-1,1) for i in range(n)]
	return Normalize(x)
#*************************************************************
def RandomPointInSphere(c,r):
	r1=random.triangular(0.1,r)
	v=RandomVector(len(c))
	return [(c[i]+v[i]*r1) for i in range(len(c))]
#										Gradient Descent
#*************************************************************
def MapRound(x,n):
	y=[]
	for i in x:
		y.append(round(i,n))
	return y
#*************************************************************
Total_Class=0
USE_CLASS_AS_WEIGHT=0

def Set_Total_Class(A):
	global Total_Class
	X=[x[len(x)-1] for x in A]
	X=map(abs,X)
	X=[(i+1) for i in X]
	Total_Class=sum(X)+0.0
#*************************************************************
Max_Iter=20
step_div=1.1
start_=1
FIXED=0
USE_BEST=1
#*************************************************************
def Gradient_Descent(f,ndim,tol=0.01,N=5000):
	step=start_
	iter_=0
	x=[0.0]*ndim
	i=0
	k=0
	j=0
	global sol
	global best
	best=9e999
	if len(sol)!=ndim:
		x=[-0.5]*ndim
	else:
		x=[i for i in sol]
	g1=0
	v=9e999
	while True:
		iter_+=1
		if iter_>=Max_Iter:
			iter_=0
			x=sol
			step=step-(step/step_div)
			if step<0.0001:
				step=0.0001
		v=f(x)
		if v<best:
			iter_=0
			j=0
			print('*GD Current Best >> '+str(v))
			i+=1
			best=v
			sol=x
		if (i>=N)or(v<=tol):
			break
		g1=Gradient1(f,x)
		g1=Normalize(g1)
		x=IncrPoint(x,g1,step*-1)
	return sol
#*************************************************************
def Reset_Data():
	f_='c:/RandOpt_'+GetDate()
	f0=open(f_,'w')
	f0.close()
#*************************************************************
def RandOpt(f,ndim,tol=1e-10,iterations=500):
	f_='c:/RandOpt_'+GetDate()
	global sol
	global best
	global OptimumReached
	global start_
	OptimumReached=0
	r=start_
	iter_=0
	kk=0
	if(len(sol)==0):
		for i in range(ndim):
			sol.append(0.0)
	i=0
	while True:
		if USE_BEST:
			start_=min(best,10)
		iter_+=1
		if (FIXED==0) and iter_>=Max_Iter:
			iter_=0
			r=r-(r/step_div)
			r=round(r,4)
			if r==0.0:
				r=start_
		if i>=iterations:
			break
		if(best<=tol):
			print('Desired Optimum Reached')
			OptimumReached=1
			break
		o=RandomPointInSphere(sol,r)
		v=f(o)
		if(v<best):
			best=v
			sol=o
			iter_=0
			f0=open(f_,'a')
			f0.write(GetTime()+'\n')
			f0.write('Error : '+str(best)+'\n')
			f0.write(str(sol)+'\n')
			f0.write('................................'+'\n')
			f0.close()
			print('*Current Best ('+str(v)+')...'+GetTime())
			i+=1
#*************************************************************
def RandOpt_Simple(D,iterations=500):
	f_='c:/RandOpt_'+GetDate()
	tol=0
	global sol
	global best
	global A
	global OptimumReached
	A=D
	ndim=len(A[0])
	OptimumReached=0
	r=start_
	iter_=0
	kk=0
	if(len(sol)==0):
		for i in range(ndim):
			sol.append(0.0)
	i=0
	while True:
		iter_+=1
		if iter_>=Max_Iter:
			iter_=0
			r=r-(r/step_div)
			if r<0.000001:
				r=0.000001
		if(i>=iterations):
			break
		if(best<=tol):
			print('Desired Optimum Reached')
			OptimumReached=1
			break
		o=RandomPointInSphere(sol,r)
		v=Error_Simple_Dot_Offset(o)
		if(v<best):
			iter_=0
			best=v
			sol=o
			f0=open(f_,'a')
			f0.write(GetTime()+'\n')
			f0.write('Error : '+str(best)+'\n')
			f0.write(str(sol)+'\n')
			f0.write('................................'+'\n')
			f0.close()
			print('*Current Best ('+str(v)+')...'+GetTime())
			i+=1
#*************************************************************
def RandOpt_Simple_NoBias(D,iterations=500):
	f_='c:/RandOpt_'+GetDate()
	tol=0
	global sol
	global best
	global A
	global OptimumReached
	A=D
	ndim=len(A[0])-1
	OptimumReached=0
	r=start_
	iter_=0
	kk=0
	if(len(sol)==0):
		for i in range(ndim):
			sol.append(0.0)
	i=0
	while True:
		iter_+=1
		if iter_>=Max_Iter:
			iter_=0
			r=r-(r/step_div)
			if r<0.000001:
				r=0.000001
		if(i>=iterations):
			break
		if(best<=tol):
			print('Desired Optimum Reached')
			OptimumReached=1
			break
		o=RandomPointInSphere(sol,r)
		v=Error_Simple_Dot_Offset(o)
		if(v<best):
			iter_=0
			best=v
			sol=o
			f0=open(f_,'a')
			f0.write(GetTime()+'\n')
			f0.write('Error : '+str(best)+'\n')
			f0.write(str(sol)+'\n')
			f0.write('................................'+'\n')
			f0.close()
			print('*Current Best ('+str(v)+')...'+GetTime())
			i+=1
#*************************************************************
#										Split Vector
#*************************************************************
def SplitVector(x,k=1):
	l=len(x)-k
	i=x[:l]
	j=x[l:]
	if(k==1):
		return [i,j[0]]
	return [i,j]
#*************************************************************
# split array into n parts
def Split(A,n):
	B=[]
	k=0
	while True:
		if k>=len(A)-1:
			break
		x=A[k:k+n]
		B.append(x)
		i=A.index(x[len(x)-1])
		k=i+1
	return B
#*************************************************************
#										Dot
#*************************************************************
def Dot(A,B):
	return sum(it.imap(operator.mul,A,B))
#*************************************************************
def Randn(n,a=-10,b=10):
	x=[]
	for i in range(n):
		x.append(random.triangular(a,b))
	return x
#*************************************************************
def Initn(k):
	return [0 for i in range(k)]
#*************************************************************
def Sigmoid(x):
	try:
		return 1./(1+2.718281828459045**-x)
	except:
		return 0
#*************************************************************
def Tanh(x):
	y=2.718281828459045**(2*x)
	return (y-1)/(y+1)
#*************************************************************
def LinearTF(x):
	return x
#*************************************************************
#TF_=Sigmoid
TF_=LinearTF
#TF_=Tanh
OUTPUT_TF_=LinearTF
BIAS=0
#*************************************************************
#										Deep Neural Network
#*************************************************************
class DeepNeuralNetwork:
	#*********************************************
	def __init__(self,A_=[1,2,1],bias_=True):
		#***************************
		# constrain to 1 or more layers
		A=A_
		if(len(A)<2):
			A=[1,2]
		#***************************
		x0=A[0] # number of inputs
		x1=A[1:len(A)-1] # layers
		x2=A[len(A)-1] # number of output
		self.nInputs=x0
		self.nOutputs=x2
		self.nLayers=len(x1)
		self.weights=[]
		self.Architecture=A
		self.bias=bias_
		#***************************
		# weight count
		y=0
		for i in range(0,len(A)-1):
			y+=A[i]*A[i+1]
			# add bias
			if self.bias:
				y+=A[i+1]
		self.weightCount=y
		#***************************
		# if 0 layers
		if(len(x1)==0):
			# weight for output
			layer_weights=[]
			for j in range(x2):
				# plus bias
				if self.bias:
					v=Randn(x0+1)
				else:
					v=Randn(x0)
				layer_weights.append(v)
			self.outputWeights=layer_weights
		#***************************
		# if 1 or more layers
		else:
			W=[]
			# for each layer
			# l is number of nodes for layer
			for i in range(1,len(A)-1):
				k=A[i-1] # number of nodes in previous layer
				l=A[i]
				layer_weights=[]
				# create n nodes for layer
				# store weight for each node of the layer
				for j in range(l):
					# plus bias
					if self.bias:
						v=Randn(k+1)
					else:
						v=Randn(k)
					layer_weights.append(v)
				W.append(layer_weights)
			self.weights=W
			#***************************
			# weight for output
			x0=A[len(A)-2]
			layer_weights=[]
			for j in range(x2):
				# plus bias
				if self.bias:
					v=Randn(x0+1)
				else:
					v=Randn(x0)
				layer_weights.append(v)
			self.outputWeights=layer_weights
	#*********************************************
	def GetWeights(self):
		v=[]
		for i in range(self.nLayers):
			layer_weights=self.weights[i]
			# for each node weight
			for x in layer_weights:
				for y in x:
					v.append(y)
		# add output weights
		weights=self.outputWeights
		for x in weights:
			for y in x:
				v.append(y)
		return v
	#*********************************************
	def SetWeights(self,W):
		# if 0 layers
		k=0
		n=len(self.outputWeights[0])
		if(self.nLayers==0):
			for i in range((self.nOutputs)):
				for j in range(n):
					self.outputWeights[i][j]=W[k]
					k+=1
		# if 1 or more layers
		else:
			# for each layer
			for i in range(self.nLayers):
				# for each node
				for j in range(len(self.weights[i])):
					for u in range(len(self.weights[i][j])):
						self.weights[i][j][u]=W[k]
						k+=1
			# set output weight
			n=len(self.outputWeights[0])
			for i in range((self.nOutputs)):
				for j in range(n):
					self.outputWeights[i][j]=W[k]
					k+=1
	#*********************************************
	def ComputeNetworkOutput(self,input_,TF=TF_):
		# if 0 layers
		if(self.nLayers==0):
			v=[]
			weights=self.outputWeights
			for i in range(self.nOutputs):
				if self.bias:
					weight_=weights[i][:len(weights[i])-1]
					bias_=weights[i][len(weights[i])-1]
					x=OUTPUT_TF_(Dot(input_,weight_)+bias_)
					v.append(x)
				else:
					x=OUTPUT_TF_(Dot(input_,weights[i]))
					v.append(x)
			return v
		# if 1 or more layers
		else:
			currentInput=input_
			for i in range(self.nLayers):
				v=[]
				for j in range(len(self.weights[i])):
					if self.bias:
						weight_=self.weights[i][j][:len(self.weights[i][j])-1]
						bias_=self.weights[i][j][len(self.weights[i][j])-1]
						x=TF(Dot(weight_,currentInput)+bias_)
						v.append(x)
					else:
						weight_=self.weights[i][j]
						x=TF(Dot(weight_,currentInput))
						v.append(x)
				currentInput=v
			# final output
			z=[]
			for i in range(self.nOutputs):
				if self.bias:
					weight_=self.outputWeights[i][:len(self.outputWeights[i])-1]
					bias_=self.outputWeights[i][len(self.outputWeights[i])-1]
					x=OUTPUT_TF_(Dot(weight_,currentInput)+bias_)
					z.append(x)
				else:
					x=OUTPUT_TF_(Dot(self.outputWeights[i],currentInput))
					z.append(x)
			return z
#*************************************************************
def CreateDeepNeuralNetwork(A=[2,1]):
	X=DeepNeuralNetwork(A)
	return X
#*************************************************************
def TrimSol(n=3):
	global sol
	sol=[round(i,n) for i in sol]
#*************************************************************
def Error(X):
	if(NN.nOutputs==1):
		s=0.0
		for p in A:
			v0,v1=SplitVector(p,NN.nOutputs)
			NN.SetWeights(X)
			y=NN.ComputeNetworkOutput(v0)[0]
			if USE_CLASS_AS_WEIGHT:
				s+=((v1-y)**2)*((abs(v1))/Total_Class)
			else:
				s+=((v1-y)**2)
		if USE_CLASS_AS_WEIGHT:
			return s
		else:
			return (s/len(A))
	else:
		s=0.0
		for p in A:
			v0,v1=SplitVector(p,NN.nOutputs)
			NN.SetWeights(X)
			y=NN.ComputeNetworkOutput(v0)
			s+=Distance(v1,y)
		return (s/len(A))
#*************************************************************
def Error_Simple_Dot(X):
	s=0.0
	for p in A:
		n=len(p)
		v1=p[:n-1]
		v2=p[n-1]
		v=Dot(X,v1)
		s+=(v-v2)**2
	return (s/len(A))
#*************************************************************
def Error_Simple_Dot_Offset(X):
	s=0.0
	for p in A:
		n1=len(p)
		n2=len(X)
		v1=p[:n1-1]
		v2=p[n1-1]
		X1=X[:n2-1]
		X2=X[n2-1]
		v=Dot(X1,v1)+X2
		s+=(v-v2)**2
	return (s/len(A))
#*************************************************************
# W weight for each point in A
W=[]
def Error_Weights(X):
	s=0.0
	i=-1
	if(NN.nOutputs==1):
		for p in A:
			i+=1
			v0,v1=SplitVector(p,NN.nOutputs)
			NN.SetWeights(X)
			y=NN.ComputeNetworkOutput(v0)[0]
			s+=W[i]*((v1-y)**2)
	else:
		for p in A:
			i+=1
			v0,v1=SplitVector(p,NN.nOutputs)
			NN.SetWeights(X)
			y=NN.ComputeNetworkOutput(v0)
			s+=W[i]*Distance(v1,y)
	return s
#*************************************************************
# Find_Min(Error,Randn(NN.weightCount))
#*************************************************************
def Pop(D0,N=100,n=10):
	for i in range(N):
		GetNetworkWeights_RandOpt(D0,iterations=n)
		if OptimumReached:
			break
	winsound.Beep(50,500)
#*************************************************************
def GetNetworkWeights_RandOpt(Data=[],Architecture=[1],tol=0.001,iterations=100000,PrintSol=0):
	global TF_
	global NN
	global A
	global best
	global sol
	global A_W
	A=Data
	Set_Total_Class(A)
	Reset_Data()
	Architecture=[len(Data[0])-1]+Architecture
	NN=DeepNeuralNetwork(Architecture,BIAS)
	RandOpt(Error,NN.weightCount,tol,iterations)
	if PrintSol:
		print('Error >> '+str(best))
		print('Weights >> '+str(sol))
	NN.SetWeights(sol)
#*************************************************************
def GetNetworkWeights_Find_Min(Data=[],Architecture=[1],Transfer_Function=0,tol=0.001,iterations=100000):
	global TF_
	global NN
	global A
	global best
	global sol
	#best=9e999
	#sol=[]
	if Transfer_Function==0:
		TF_=LinearTF
	if Transfer_Function==1:
		TF_=Sigmoid
	if Transfer_Function==2:
		TF_=Tanh
	A=Data
	Architecture=[len(Data[0])-1]+Architecture
	NN=DeepNeuralNetwork(Architecture)
	Find_Min(Error,Randn(NN.weightCount),tol=tol,N=iterations)
	print('Error >> '+str(best))
	print('Weights >> '+str(sol))
	NN.SetWeights(sol)
	return sol
#*************************************************************
def GetNetworkWeights_Gradient_Descent(Data=[],Architecture=[1],tol=0.001,iterations=100000):
	global TF_
	global NN
	global A
	global best
	global sol
	A=Data
	Architecture=[len(Data[0])-1]+Architecture
	NN=DeepNeuralNetwork(Architecture)
	sol=Gradient_Descent(Error,NN.weightCount,tol=tol,N=iterations)
	NN.SetWeights(sol)
#*************************************************************
def ComputeNetworkOutput(input_):
	try:
		x=NN.ComputeNetworkOutput(input_,TF_)
		if(len(x)==1):
			return x[0]
		return x
	except:
		return None
#*************************************************************
def GetPointOnly(x):
	i=len(x)
	return x[:i-1]
#*************************************************************
def GetClassOnly(x):
	i=len(x)
	return x[i-1]
#*************************************************************
def ComputeOutputSImple(p):
	X=sol
	X1=GetPointOnly(X)
	X2=GetClassOnly(X)
	return Dot(X1,p)+X2
#*************************************************************
def ComputeOutputSImple_NoBias(p):
	return Dot(sol,p)
#*************************************************************
div=10
min_=None
max_=None
USE_INT=1
INT_STEP=1
ADD_BOUNDARY=0
#*************************************************************
def GetMinMax(D,k=1):
	ndim=len(D[0])-k
	global min_
	global max_
	min_=[9e999]*ndim
	max_=[-9e999]*ndim
	for d in D:
		for i in range(ndim):
			min_[i]=min(min_[i],d[i])
			max_[i]=max(max_[i],d[i])
#*************************************************************
TOKENS=[]
TOKENS_STR=[]
#*************************************************************
class Token:
	def __init__(self):
		# 1 >=
		# 0 <
		self.dim=0
		self.dir=1
		self.val=0
#*************************************************************
def GetToken_str(t):
	if t.dir:
		return 'x_'+str(t.dim)+'>='+str(t.val)
	else:
		return 'x_'+str(t.dim)+'<'+str(t.val)
#*************************************************************
# t = [t1]
def GetToken_str_arr(t):
	return [GetToken_str(x) for x in t]
#*************************************************************
TOKEN_RANDOM_SELECT=1

def InitTokens(D):
	global TOKENS
	global TOKENS_STR
	TOKENS=[]
	TOKENS_STR=[]
	Tokens_1=[]
	Tokens_2=[]
	GetMinMax(D)
	d=D[0]
	# 1 boundary
	for i in range(len(d)-1):
		x0=rg.range1(min_[i],max_[i],div)
		if USE_INT:
			x0=rg.range2(min_[i],max_[i],INT_STEP)
		for j in x0:
			t=Token()
			t.dim=i
			t.dir=1
			t.val=j
			Tokens_1+=[[t]]
			t=Token()
			t.dim=i
			t.dir=0
			t.val=j
			Tokens_1+=[[t]]
	# 2 boundary
	if ADD_BOUNDARY:
		for i in range(len(Tokens_1)):
			for j in range(len(Tokens_1)):
				if j>i:
					t1=Tokens_1[i][0]
					t2=Tokens_1[j][0]
					if t1.dim==t2.dim and t1.dir==t2.dir:
						continue
					if TOKEN_RANDOM_SELECT:
						if t1.dim==t2.dim:
							k=random.choice([0]*10+[1]*1)
							if k:
								continue
						else:
							k=random.choice([0]*1+[1]*5)
							if k:
								continue
					if t1.dir==1:
						if t1.dim==t2.dim:
							if t1.val<t2.val:
								t3=[t1,t2]
								Tokens_2+=[t3]
						else:
							t3=[t1,t2]
							Tokens_2+=[t3]
					else:
						if t1.dim==t2.dim:
							if t1.val>t2.val:
								t3=[t1,t2]
								Tokens_2+=[t3]
						else:
							t3=[t1,t2]
							Tokens_2+=[t3]
	TOKENS=Tokens_1+Tokens_2
	for t in TOKENS:
		ts=GetToken_str_arr(t)
		ts=' and '.join(ts)
		ts='('+ts+')'
		TOKENS_STR.append(ts)
#*************************************************************
def TestToken(t,x):
	if t.dir:
		return int(x[t.dim]>=t.val)
	else:
		return int(x[t.dim]<t.val)
#*************************************************************
# t= [t1,t2]
def TestToken_arr(t,x):
	for t0 in t:
		k=TestToken(t0,x)
		if k==0:
			return 0
	return 1
#*************************************************************
def GetVector(x):
	A=[]
	for t in TOKENS:
		A+=[TestToken_arr(t,x)]
	return A+[x[len(x)-1]]
'''
def _GetVector_(x,trimClass=1):
	# final
	A=[]
	X=[]
	ndim=len(x)
	if trimClass:
		ndim=len(x)-1
	for i in range(ndim):
		# 1 boundary
		A0=[]
		x0=rg.range1(min_[i],max_[i],div)
		if USE_INT:
			x0=rg.range2(min_[i],max_[i],INT_STEP)
		for j in x0:
			A0+=[int(x[i]>=j)]
			A0+=[int(x[i]<j)]
		A+=A0
		if ADD_BOUNDARY:
			X.append(A0)
	# 2 boundary
	if ADD_BOUNDARY:
		A2=[]
		for i1 in range(len(X)):
			for j1 in range(len(X)):
				if j1>i1:
					X0=X[i1]
					X1=X[j1]
					for x0 in X0:
						for x1 in X1:
							A2.append(x0 & x1)
		A+=A2
	if trimClass:
		return A+[x[ndim]]
	return A
'''
#*************************************************************
def Get_D0(D):
	Reset()
	InitTokens(D)
	return [GetVector(x) for x in D]
'''
def _Get_D0_(D):
	Reset()
	GetMinMax(D)
	return [_GetVector_(x) for x in D]
'''
#*************************************************************
'''
def GetVector_str(x):
	A=[]
	X=[]
	ndim=len(x)
	for i in range(ndim):
		# 1 boundary
		A0=[]
		x0=rg.range1(min_[i],max_[i],div)
		if USE_INT:
			x0=rg.range2(min_[i],max_[i],INT_STEP)
		for j in x0:
			A0.append('x_'+str(i)+'>='+str(j))
			A0.append('x_'+str(i)+'<'+str(j))
		A+=A0
		if ADD_BOUNDARY:
			X.append(A0)
	# 2 boundary
	if ADD_BOUNDARY:
		A2=[]
		for i1 in range(len(X)):
			for j1 in range(len(X)):
				if j1>i1:
					X0=X[i1]
					X1=X[j1]
					for x0 in X0:
						for x1 in X1:
							A2.append('('+x0+' and '+x1+')')
		A+=A2
	return A
'''
#*************************************************************
def PrintVar(D,A='',prefix=''):
	# returns 'X[0],X[1],X[2]'
	def GetX(N):
		x=''
		for i in range(N):
			x+='X['+str(i)+'],'
		return x[:len(x)-1]
	#
	def Fix_(G,x):
		if 'and' in x:
			x1=x
			x1=x1.replace('(','')
			x1=x1.replace(')','')
			a=x1.split(' and ')
			b=[Fix_(G,y) for y in a]
			b=' and '.join(b)
			return b
		else:
			x1=x
			x1=x1.replace('(','')
			x1=x1.replace(')','')
			x_i=''
			if '>' in x1:
				x_i=x1[:x1.index('>')]
			elif '<' in x1:
				x_i=x1[:x1.index('<')]
			y_=''
			x0=''
			for y in G:
				x0=G[y]
				if x0==x_i:
					y_=y
					break
			x1=x1.replace(x0,y_)
			return x1
	#
	if A=='':
		A=GetX(len(D[0])-1)
	A=A.split(',')
	G={}
	for i in range(len(A)):
		G[A[i]]='x_'+str(i)
	TrimSol()
	_x_=''
	_x_+=prefix+'('
	for i in range(len(sol)):
		x=TOKENS_STR[i]
		x=Fix_(G,x)
		_x_+='(int('+x+')*'+str(sol[i])+')+'
	_x_=_x_[:len(_x_)-1]
	_x_+=')'
	f0='c:/_pop_.txt'
	f00=open(f0,'w')
	f00.write(_x_+'\n')
	f00.close()
'''
def _PrintVar_(D,A='',prefix=''):
	# returns 'X[0],X[1],X[2]'
	def GetX(N):
		x=''
		for i in range(N):
			x+='X['+str(i)+'],'
		return x[:len(x)-1]
	#
	def Fix_(G,x):
		if 'and' in x:
			x1=x
			x1=x1.replace('(','')
			x1=x1.replace(')','')
			a=x1.split(' and ')
			b=[Fix_(G,y) for y in a]
			b=' and '.join(b)
			return b
		else:
			x1=x
			x_i=''
			if '>' in x1:
				x_i=x1[:x1.index('>')]
			elif '<' in x1:
				x_i=x1[:x1.index('<')]
			y_=''
			x0=''
			for y in G:
				x0=G[y]
				if x0==x_i:
					y_=y
					break
			x1=x1.replace(x0,y_)
			return x1
	#
	if A=='':
		A=GetX(len(D[0])-1)
	A=A.split(',')
	G={}
	for i in range(len(A)):
		G[A[i]]='x_'+str(i)
	TrimSol()
	_x_=''
	D00=GetVector_str(D[0][:len(D[0])-1])
	_x_+=prefix+'('
	for i in range(len(sol)):
		x=D00[i]
		x=Fix_(G,x)
		_x_+='(int('+x+')*'+str(sol[i])+')+'
	_x_=_x_[:len(_x_)-1]
	_x_+=')'
	f0='c:/_pop_.txt'
	f00=open(f0,'w')
	f00.write(_x_+'\n')
	f00.close()
'''
#*************************************************************
def GetPointOnly(p):
	i=len(p)-1
	return p[:i]
#*************************************************************