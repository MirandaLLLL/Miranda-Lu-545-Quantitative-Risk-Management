import numpy as np
import pandas as pd
from scipy.stats import norm, t
import scipy.optimize as optimize

# problem1

# Fit a Normal Distribution 
def norm_ES(x,alpha):
    miu = x.mean()
    x_adj=x-miu
    sigma = x_adj.std()
    norm_VaR = -norm.ppf(alpha,loc=miu,scale=sigma)
    
   
    ES_norm =- miu + sigma*norm.pdf(norm.ppf(alpha))/alpha
    print("When alpha = "+str(alpha)+" :")
    print("The VaR fitted a Normal Distribution is "+str(norm_VaR))
    print("The Expected Shortfall is "+str(ES_norm))
   
    return norm_VaR, ES_norm

# t distribution:
def MLE_t(pars, x):
    df = pars[0]
    loc=pars[1]
    scale = pars[2]
    ll = np.log(t.pdf(x, df=df,loc=loc,scale=scale)) 
    return -ll.sum()


def t_ES(x, alpha):
    mean_x=x.mean()
    std_x=x.std()
    cons = ({'type': 'ineq', 'fun': lambda x:  x[0] - 2}, {'type': 'ineq', 'fun': lambda x:  x[2]})
    model = optimize.minimize(fun = MLE_t,  x0 = [2, mean_x,std_x ], constraints=cons, args =x).x
    
    df=model[0]
    loc=model[1]
    scale=model[2]
    t_sample = t.rvs(df =df, loc = loc, scale = scale, size = 10000)
    
    t_VaR=-t.ppf(alpha, df =df, loc = loc, scale = scale)
    
    ES_t=-t_sample[t_sample<-t_VaR].mean()
    
    print("When alpha = "+str(alpha)+" :")
    print("The VaR fitted a T Distribution is "+str(t_VaR))
    print("The Expected Shortfall is "+str(ES_t))
    
    return  df,loc,scale,t_VaR,ES_t



    