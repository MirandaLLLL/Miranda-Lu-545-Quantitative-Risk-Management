from scipy.stats import norm,t
from scipy.optimize import minimize
import pandas as pd
import numpy as np
import statsmodels as sm

def return_calculate(method,price,date):
    df1=price.drop(columns=date)
    if method=="Brownian_Motion":
        return_df=df1-df1.shift()
    if method=="ArithmeticReturn":
        return_df=(df1-df1.shift())/df1.shift()
    if method=="Geometric_Brownian_Motion":
        tmp=df1/df1.shift()
        return_df=np.log(tmp)
        
    return return_df

# 1. Using a normal distribution.
def Norm_VaR(price,miu,alpha):
    sigma=price.std()
    z_score = norm.ppf(alpha,loc=miu,scale=sigma)
    VaR=-z_score
    return VaR

# 2. Using a normal distribution with an Exponentially Weighted variance (λ = 0. 94)
def Cal_weight(lamda,n):
    w=np.zeros(n)
    total_w=0
    for i in range(n):
        tmp=(1-lamda)*pow(lamda,i-1)
        w[i]=tmp
        total_w+=tmp

    w=w/total_w
    return w
    
def Cal_cov(w,x,y):
    n=len(x)
    cov=0
    x_mean=np.mean(x)
    y_mean=np.mean(y)
    
    for i in range(n):
        cov+=(x[i]-x_mean)*(y[i]-y_mean)*w[n-1-i]
    
    return cov


def EW_VaR(price,miu,alpha,lamda):
    nsize=len(price)
    weights=Cal_weight(lamda,nsize)
    
    
    sigma=np.sqrt(Cal_cov(weights,price,price))
    z_score = norm.ppf(alpha,loc=miu,scale=sigma)
    VaR=-z_score
    
    return VaR

# 3. Using a MLE fitted T distribution.

def MLE_t(pars, x):
    df = pars[0]
    sigma=pars[1]
    ll = t.logpdf(x, df=df,scale=sigma)
    return -ll.sum()

def MLE_T_VaR(price,miu,alpha):
    cons = ({'type': 'ineq', 'fun': lambda x: x[1] - 0})
    # params=t.fit(price)
    
    model = minimize(MLE_t, [price.size, 1], args = price, constraints = cons)
    estimator=model.x
    VaR = -t.ppf(alpha, df=estimator[0], loc=miu, scale=estimator[1])
    
    return VaR

# 4. Using a fitted AR(1) model.

def AR_VaR(price,miu,alpha):
    price=np.array(price)
    model = sm.tsa.ar_model.AutoReg(price, lags = 1)
    results = model.fit()
    a = results.params[0]
    beta = results.params[1]
    sigma = results.resid.std()
    
    z_score = norm.ppf(alpha,loc=0,scale=1)
    Y_t = price[-1]
    VaR = -((a + beta*Y_t)+z_score*sigma)
    return VaR

# 5. Using a Historic Simulation.

def Historic_Simulation(price,alpha,N_draws):
    
    simulate_list = price.sample(N_draws, replace=True)
    VaR=-np.percentile(simulate_list,(100 * alpha))
    
    return VaR