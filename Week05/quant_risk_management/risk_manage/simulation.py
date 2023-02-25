import numpy as np

# PCA
def PCA(cov_mat):
    
    vals,vecs =np.linalg.eigh(cov_mat)
    tv=np.sum(vals)
    
    explain_total=0
    explain_list=[]
    
    for i in vals:
        if i > 0:
            explain_total+=i
            explain_list.append(i)
            
    
   
    exp_list=sorted(explain_list,reverse=True)
    exp_cum_list=np.cumsum(exp_list)
    exp_cum_list=np.divide(exp_cum_list,explain_total)
    
    return exp_cum_list

#Direct Simulate
def chol_psd(root,a):
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



def direct_simulate(cov_mat, size ):
    n=len(cov_mat)
    root=np.zeros((n,n))
    Gen_rand = np.random.normal(size = (n,size))
    root = chol_psd(root,cov_mat)
    ds=np.matmul(root,Gen_rand)
    return ds

# PCA simulate

def PCA_simulate(cov_mat, size, nval):
   
    vals, vecs= np.linalg.eigh(cov_mat)
    exp_list = PCA(cov_mat)
    
    n = len(cov_mat)

    for i in range(n):
        if exp_list[i]>=nval: 
            index = i
            break
        
    vals = vals[(n-index-1):]
    vecs = vecs[:, (n-index-1):]
    
    B = np.matmul(vecs, np.diag(np.sqrt(vals)))
    r = np.random.normal(size = (len(vals),size))
    ps=np.matmul(B,r)
    return ps


def generate_Classical_Brownian_Motion(sigma,t,p0):
    p=[]
    r=[]
    p.append(p0)
    
    for i in range(t):
        rt=np.random.normal(0,sigma)
        r.append(rt)
        tmp=p[i]+rt
        p.append(tmp)
    
    # print("The Classical Brownian Motion's result:")
    # print("Mean:"+ str(np.mean(p)))
    # print("Standard deviation:"+ str(np.std(p)))
    
    return p,np.mean(p),np.std(p)
    
def generate_Arithmetic_Return_System(sigma,t,p0):
    p=[]
    r=[]
    p.append(p0)
    
    for i in range(t):
        rt=np.random.normal(0,sigma)
        r.append(rt)
        tmp=p[i]*(1+rt)
        p.append(tmp)
    
    # print("The Arithmetic Return System's result:")
    # print("Mean:"+ str(np.mean(p)))
    # print("Standard deviation:"+ str(np.std(p)))
    
    return p,np.mean(p),np.std(p)

def generate_Geometric_Brownian_Motion(sigma,t,p0):
    p=[]
    r=[]
    p.append(p0)
    
    for i in range(t):
        rt=np.random.normal(0,sigma)
        r.append(rt)
        tmp=p[i]*np.exp(rt)
        p.append(tmp)
    
    # print("The Geometric Brownian Motion's result:")
    # print("Mean:"+ str(np.mean(p)))
    # print("Standard deviation:"+ str(np.std(p)))
    
    return p,np.mean(p),np.std(p)

def test_std(n,sigma,p0,t):
    p1=[]
    p2=[]
    p3=[]
    
    for i in range(n):
        p1.append(generate_Classical_Brownian_Motion(sigma,t,p0)[0][t-1])
        p2.append(generate_Arithmetic_Return_System(sigma,t,p0)[0][t-1])
        p3.append(generate_Geometric_Brownian_Motion(sigma,t,p0)[0][t-1])
        
    mean1=np.mean(p1)
    mean2=np.mean(p2)
    mean3=np.mean(p3)
    std1=np.std(p1)
    std2=np.std(np.log(p2))
    std3=np.std(np.log(p3))
    
    print('When sigma = '+str(sigma)+' , and t = '+str(t)+' , and p0 = '+str(p0)+" :")
    print("After "+str(n)+" times' generating:")
    print("Classical_Brownian_Motion:")
    print("                   mean: "+str(mean1))
    print("                   standard deviation:"+str(std1))
    print("                   Expected mean: "+str(p0))
    print("                   Expected standard deviation:"+str(np.sqrt(t)*sigma))
    print("Arithmetic_Return_System:")
    print("                   mean: "+str(mean2))
    print("                   standard deviation:"+str(std2))
    print("                   Expected mean: "+str(p0))
    print("                   Expected standard deviation:"+str(np.sqrt(t)*sigma))
    print("Geometric_Brownian_Motion:")
    print("                   mean: "+str(mean3))
    print("                   standard deviation:"+str(std3))
    print("                   Expected mean: "+str(p0))
    print("                   Expected standard deviation:"+str(np.sqrt(t)*sigma))
        