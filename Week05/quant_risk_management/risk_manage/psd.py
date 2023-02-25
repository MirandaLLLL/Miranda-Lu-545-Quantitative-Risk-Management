import numpy as np
import pandas as pd 

# implement chol_psd 
def chol_psd(a):
    n=len(a)
    
    root=np.zeros((n,n))
    
    for j in range(n):
        if j==0:
            s=0
        else:
            s=np.matmul(root[j,:j],root[j,:j].T)
    
        temp=a[j,j]-s
        if 0 >= temp >= -1e-8:
            temp = 0.0
            
        root[j,j]=np.sqrt(temp)
        
        if root[j,j]==0:
            continue
        
        ir=1.0/root[j,j]
        
        for i in range((j+1),n):
            s=np.matmul(root[i,:j],root[j,:j]) 
            root[i,j]=(a[i,j]-s)*ir
                      
    return root

#implement near_psd:

def near_psd(a,epsilon=0.0):
    
    n=len(a)
    
    invSD=np.array([])
    out=a.copy()
    
    if np.count_nonzero(np.diag(a)==1) !=n:
        invSD=np.diagflat(1/np.sqrt(np.diag(out)))
        tmp=np.matmul(out,invSD)
        out=np.matmul(invSD,tmp)
        
    vals,vecs=np.linalg.eigh(a)
    vals=np.maximum(vals,epsilon)
    temp=np.matmul(vecs,vecs)
    T=1/np.matmul(vecs*vecs,vals)
    T=np.diagflat(np.sqrt(T))
    l=np.diagflat(np.sqrt(vals))
    tmp2=np.matmul(T,vecs)
    B=np.matmul(tmp2,l)
    out=np.matmul(B,B.T)
    
    if len(invSD) != 0:
        invSD = np.diagflat(1/np.diag(invSD))
        tmp3=np.matmul(out,invSD)
        out=np.matmul(invSD,tmp3)
        
    return out
# Implement Higham 2002


def ProjectionU(A):
    p=A.copy()
    
    for i in range(len(A)):
       p[i,i]=1
    return p

def ProjectionS(A,W):
    w_sqrt=np.sqrt(W)
    tmp=np.matmul(w_sqrt,A)
    
    tmp2=np.matmul(tmp,w_sqrt)
    
    vals , vecs = np.linalg.eigh(A)
    vals = np.maximum(vals,0)
    val_diag= np.diagflat(vals)
    
    tmp=np.matmul(vecs,val_diag)
    p=np.matmul(tmp,vecs.T)
    
    return p

def Frobenius_Norm(A):
    tot=0
    for i in range(len(A)):
        for j in range(len(A)):
            tot+=A[i,j]**2
            
    return tot


def Higham_2002(A,iteration=1000):
    Lag_deltaS = 0
   
    Lag_Y = A
    
    Lag_Gamma = np.inf
    
    weights=np.ones(len(A))
    
    tol = 1e-10
    
    for i in range(iteration):
        R = Lag_Y - Lag_deltaS
        X=ProjectionS(R,weights)
        deltaS=X-R
        Y=ProjectionU(X)
        Gamma=Frobenius_Norm(Y-A)
        
        if abs(Gamma - Lag_Gamma)<tol:
            break
        
        Lag_deltaS=deltaS
        Lag_Y=Y
        Lag_Gamma=Gamma
        
    return Y
    

