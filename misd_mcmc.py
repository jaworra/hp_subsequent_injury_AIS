#from re import I
import random 
import math
import numpy as np

import scipy.stats as stats
from scipy.stats import multinomial
from scipy.stats import multivariate_normal
from scipy.stats import gamma
from scipy.stats import expon
from scipy.stats import uniform
from scipy.special import expit
from numpy.polynomial import legendre



def uv_para_em_fit(points_hawkes,T,max_time_phi,num_iter=500,reltol=1e-5):
    """`
    Implements Time scale separation and EM convergence of univariate Hawkes process
    with exponential decay (Lewis & Mohler, 2011). 

    :param points_hawkes: Bounded finite sample of the process up to time T. 1-d ndarray of timestamps
    :param T: the maximum time
    :param max_time_phi: maximum time on kernel domain
    :param num_iter: int, maximum number of EM iterations
    :param reltol: double, the relative improvement tolerance to stop the algorithm

    :return: 3 lists of parameter estimate (mu, alpha and beta)
    """
    #initialising 
    odll = 0.
    odll_p = 0.
    T_phi=max_time_phi

    N = len(points_hawkes)
    P, Pij_flat, tau_phi_md, num_Pij_row = ini_P(points_hawkes,T_phi)

    tau_phi = sum(tau_phi_md,[])
    tau_Pij_flat = [a * b for a, b in zip(tau_phi, Pij_flat)]

    #set
    lamda_mu = sum(np.diag(P))*2/T                #lamda_mu = sum(np.diag(P))/T
    lamda_alpha = sum(Pij_flat)*2/N/T_phi         #lamda_alpha = sum(Pij_flat)/N    
    lambda_beta = sum(Pij_flat)/sum(tau_Pij_flat) #lambda_beta = sum(Pij_flat)/sum(tau_phi)
    odll_p = uv_exp_ll(points_hawkes, lamda_mu, lamda_alpha, lambda_beta, T)

    mu_list=[]
    alpha_list=[] 
    beta_list=[]
    like_list=[]

    #print('update prob matrix')
    for iteration in range(num_iter): #ensure convergence
        Pij_flat=[]
        tau = []
    
        # E Step, update P 
        mu_ti = lamda_mu
        alpha_ti = lamda_alpha
        beta_ti = lambda_beta
        for j in range(N):
            sum_ij=0
            
            for i in range(j,N):  # lower triangular matrix
                if i == 0: continue ## start from the second row, because the first row of P is always 1'
                tij = points_hawkes[i] - points_hawkes[j] 

                kernel = 0
                for k in range(0,i): #calculate kernel   #for k in range(0,i-1):
                    tik = points_hawkes[i] - points_hawkes[k] 
                    kernel += (alpha_ti * beta_ti * np.exp(-beta_ti *tik))

                if i != j:
                    P[i][j] = (alpha_ti * beta_ti * np.exp(-beta_ti * tij))/(kernel + mu_ti)
                    Pij_flat.append(P[i][j])
                    tau.append(tij)
                    sum_ij += P[i][j] 
                else:
                    P[i][i] = mu_ti/(kernel + mu_ti)

                
        # M Step, update paramters (maximisation)
        tau_Pij_flat = [a * b for a, b in zip(tau, Pij_flat)] #tau_Pij_flat = [a * b for a, b in zip(tau_phi, Pij_flat)] #tau does not change
        lamda_mu = sum(np.diag(P))/T                #lamda_mu = sum(np.diag(P))*2/T                
        lamda_alpha = sum(Pij_flat)/N               #lamda_alpha = sum(Pij_flat)*2/N/T_phi 
        lambda_beta = sum(Pij_flat)/sum(tau_Pij_flat)  # is this right , #lambda_beta = sum(Pij_flat)/sum(tau_phi)
        

        # calculate observed data log likelihood
        odll = uv_exp_ll(points_hawkes, lamda_mu, lamda_alpha, lambda_beta, T)
        relimp = (odll - odll_p) / abs(odll_p)  # relative improvement
        if relimp < -1e-5:
            raise Exception("Convergence problem, the log likelihood did not increase")
        elif relimp < reltol:
            break
        odll_p = odll

        # record
        mu_list.append(lamda_mu)
        alpha_list.append(lamda_alpha)
        beta_list.append(lambda_beta)
        like_list.append(odll)

        #checks --
        #print('mu: '+str(lamda_mu)+ ', alpha: '+str(lamda_alpha)+', beta: '+str(lambda_beta))
        # all elements should equal 1
        sum_rows =[]
        for i in range(N):
            sum_rows.append(sum(P[i]))
        
    #branching structure
    B = np.zeros_like(P)
    B[np.arange(len(P)), P.argmax(1)] = 1

    return [lamda_mu, lamda_alpha,lambda_beta], mu_list,alpha_list,beta_list


def uv_exp_ll(t, mu, alpha, theta, T):
    """
    Likelihood of a univariate Hawkes process with exponential decay.

    :param t: Bounded finite sample of the process up to time T. 1-d ndarray of timestamps
    :param mu: the exogenous intensity
    :param alpha: the infectivity factor alpha
    :param theta: intensity parameter of the delay density
    :param T: the maximum time
    :return: the log likelihood
    """

    phi = 0.
    lComp = -mu * T
    lJ = 0
    N = t.shape[0]
    j = 0

    lComp -= alpha * (1 - math.exp(-theta * (T - t[0])))
    lJ = math.log(mu)

    for j in range(N-1):
        d = t[j+1] - t[j]
        r = T - t[j+1]

        ed = math.exp(-theta * d)  # exp_diff
        try:
            F = 1 - math.exp(-theta * r)
        except: # number is too large
            F = 1 - math.exp(50)

        phi = ed * (1 + phi)
        lda = mu + alpha * theta * phi

        lJ = lJ + math.log(lda)
        lComp -= alpha * F

    return lJ + lComp


def ini_P(points_hawkes,T_phi):
    r"""
    Initialize the probabilistic branching matrix. 
    :rtype: numpy array
    :return: probabilistic branching matrix, the flattened P_ij, interval of timestamps :\tau, the number of P_ij!=0 in each row 
    """
    N = len(points_hawkes)
    P = np.zeros((N,N))
    Pij_flat = []
    tau = []
    num_Pij_row = []
    for i in range(N):                    # initial value of P
        for j in range(i+1):
            tij = points_hawkes[i] - points_hawkes[j]
            if tij >= T_phi: continue
            else:
                P[i][j:i+1] = np.random.dirichlet([1]*(i-j+1))
                Pij_flat += list(P[i][j:i])
                tau.append(list(points_hawkes[i] - np.array(points_hawkes[j:i])))
                num_Pij_row.append(i-j)
    return P, Pij_flat, tau, num_Pij_row



    

def uv_para_em_mcmc_fit(points_hawkes,T,max_time_phi,num_iter=500,reltol=1e-5):
    """`
    Implements Time scale separation and EM convergence of univariate Hawkes process
    with exponential decay (Lewis & Mohler, 2011). 

    :param points_hawkes: Bounded finite sample of the process up to time T. 1-d ndarray of timestamps
    :param T: the maximum time
    :param max_time_phi: maximum time on kernel domain
    :param num_iter: int, maximum number of EM iterations
    :param reltol: double, the relative improvement tolerance to stop the algorithm

    :return: 3 lists of parameter estimate (mu, alpha and beta)
    """
    #initialising 
    odll = 0.
    odll_p = 0.
    T_phi=max_time_phi

    N = len(points_hawkes)
    P, Pij_flat, tau_phi_md, num_Pij_row = ini_P(points_hawkes,T_phi)

    tau_phi = sum(tau_phi_md,[])
    tau_Pij_flat = [a * b for a, b in zip(tau_phi, Pij_flat)]

    #set
    lamda_mu = sum(np.diag(P))*2/T                #lamda_mu = sum(np.diag(P))/T
    lamda_alpha = sum(Pij_flat)*2/N/T_phi         #lamda_alpha = sum(Pij_flat)/N    
    lambda_beta = sum(Pij_flat)/sum(tau_Pij_flat) #lambda_beta = sum(Pij_flat)/sum(tau_phi)
    odll_p = uv_exp_ll(points_hawkes, lamda_mu, lamda_alpha, lambda_beta, T)

    mu_list=[]
    alpha_list=[] 
    beta_list=[]
    like_list=[]



    #print('update prob matrix')
    for iteration in range(num_iter): #ensure convergence
        Pij_flat=[]
        tau = []
    
        # E Step, update P 
        mu_ti = lamda_mu
        alpha_ti = lamda_alpha
        beta_ti = lambda_beta
        for j in range(N):
            sum_ij=0
            
            for i in range(j,N):  # lower triangular matrix
                if i == 0: continue ## start from the second row, because the first row of P is always 1'
                tij = points_hawkes[i] - points_hawkes[j] 

                kernel = 0
                for k in range(0,i): #calculate kernel   #for k in range(0,i-1):
                    tik = points_hawkes[i] - points_hawkes[k] 
                    kernel += (alpha_ti * beta_ti * np.exp(-beta_ti *tik))

                if i != j:
                    P[i][j] = (alpha_ti * beta_ti * np.exp(-beta_ti * tij))/(kernel + mu_ti)
                    Pij_flat.append(P[i][j])
                    tau.append(tij)
                    sum_ij += P[i][j] 
                else:
                    P[i][i] = mu_ti/(kernel + mu_ti)
                    

        # M Step, update paramters (maximisation)
        tau_Pij_flat = [a * b for a, b in zip(tau, Pij_flat)] #tau_Pij_flat = [a * b for a, b in zip(tau_phi, Pij_flat)] #tau does not change
        lamda_mu = sum(np.diag(P))/T                #lamda_mu = sum(np.diag(P))*2/T                
        lamda_alpha = sum(Pij_flat)/N               #lamda_alpha = sum(Pij_flat)*2/N/T_phi 
        lambda_beta = sum(Pij_flat)/sum(tau_Pij_flat)  # is this right , #lambda_beta = sum(Pij_flat)/sum(tau_phi)
        
        #samples
        
        # calculate observed data log likelihood
        odll = uv_exp_ll(points_hawkes, lamda_mu, lamda_alpha, lambda_beta, T)
        relimp = (odll - odll_p) / abs(odll_p)  # relative improvement
        if relimp < -1e-5:
            raise Exception("Convergence problem, the log likelihood did not increase")
        elif relimp < reltol:
            break
        odll_p = odll

        # record
        mu_list.append(lamda_mu)
        alpha_list.append(lamda_alpha)
        beta_list.append(lambda_beta)
        like_list.append(odll)

        #checks --
        #print('mu: '+str(lamda_mu)+ ', alpha: '+str(lamda_alpha)+', beta: '+str(lambda_beta))
        #all elements should equal 1
        sum_rows =[]
        for i in range(N):
            sum_rows.append(sum(P[i]))
    
    

    #branching structure
    B = np.zeros_like(P)
    B[np.arange(len(P)), P.argmax(1)] = 1

    
    #check child
    diag_B = np.diag(B)

    #child
    child=[]
    parents=[]
    for k in range(N):
        if diag_B[k] == 1:
            parents.append(points_hawkes[k]) 
        else:
            child.append(points_hawkes[k])


    return [lamda_mu, lamda_alpha,lambda_beta], mu_list,alpha_list,beta_list,child,parents
