
import numpy as np
import csv
import pandas as pd
import os
import random 
import math
import timeit

import statsmodels.api as sm
import scipy.stats as stats

#for the max log likelihood
from scipy.optimize import minimize
from matplotlib import pyplot as plt
from patsy import dmatrices



def plot_parameters():
    import seaborn as sns
    
    script_dir = os.path.dirname(__file__)
    df = pd.read_csv(os.path.join(script_dir,'mle_est_hp_3digits.csv'))
    print(df.dtypes)
    
    df['Player'] = df['Player'].str.replace("Player "," - ")
    print(df)
    print(df.dtypes)
    #return

    f, axes = plt.subplots(2, 2,figsize=(10,10))
    #f, axes = plt.subplots(1, 4)
    #f, axes = plt.subplots(1, 3)
    #f, axes = plt.subplots(3, 1)

    sns.barplot(x='mu', y='Player', data=df.sort_values('mu', ascending=True), palette='Blues', ax=axes[0,0])
    plt.ylabel('Player', fontsize=12)
    plt.xlabel('mu', fontsize=12)
    plt.xticks(rotation='horizontal')

    sns.barplot(x='branch-ratio', y='Player', data=df.sort_values('branch-ratio', ascending=True), palette='Blues', ax=axes[0,1])
    plt.ylabel('Player', fontsize=12)
    plt.xlabel('branch-ratio', fontsize=12)
    plt.xticks(rotation='horizontal')

    sns.barplot(x='alpha', y='Player', data=df.sort_values('alpha', ascending=True), palette='Blues', ax=axes[1,0])
    plt.ylabel('Player', fontsize=12)
    plt.xlabel('alpha', fontsize=12)
    plt.xticks(rotation='horizontal')

    sns.barplot(x='beta', y='Player', data=df.sort_values('beta', ascending=True), palette='Blues', ax=axes[1,1])
    plt.ylabel('Player', fontsize=12)
    plt.xlabel('beta', fontsize=12)
    plt.xticks(rotation='horizontal')

    plt.show()
    return

def plot_fit(path,filename, title,smp,max_time,mu,alpha,beta,iter_neg_ll):
    '''
    Saves plots. Intensity, MLE parameter estimates and goodeness of fit
    '''

    f, axs = plt.subplots(1, 3,figsize=(20,4))
    f.suptitle(title)
    #intensity
    lda_ar = [mu + np.sum(alpha * beta * np.exp(-beta * (x - smp[smp < x]))) \
             for x in np.arange(0, max_time, .1)]

    axs[0].set_title('')
    axs[0].set_ylabel("$\lambda^*(t)$")
    axs[0].set_xlabel("$t$")
    #axs[0].plot(smp, np.ones_like(smp) * .1, 'k.',label="arrivals")
    axs[0].plot(smp, np.ones_like(smp) * min(lda_ar), 'k.',label="arrivals")
    axs[0].plot(np.arange(0, max_time, .1), lda_ar, color= '#1f77b4',label="$\lambda^*(t)$")#'b-' ##ff7f0e orange
    #axs[0].legend(loc='upper left')


    #axs[1].set_title('Parameter Estimation')
    axs[1].set_title('')
    axs[1].plot(iter_neg_ll)
    axs[1].set_ylabel('neg LL')
    axs[1].set_xlabel('iterations')


    tau =  np.empty(len(smp), dtype=np.float64)
    t_pre = 0
    for i, t_i in enumerate(smp):
        tau[i] = t_i - t_pre
        t_pre = t_i
    stats.probplot(tau, dist="expon", plot=plt)
    #axs[2].set_title('QQ Plot')
    axs[2].set_title('')
    axs[2].get_lines()[0].set_color('black') ##1f77b4
    axs[2].get_lines()[0].set_markersize(3)
    axs[2].get_lines()[1].set_color('#1f77b4')
    axs[2].set_ylabel(r'ordered $\tau$')
    axs[2].set_xlabel('theoretical quantiles')

    #output
    plt.savefig(os.path.join(path,'img\\comparison\\'+filename+'.png'), bbox_inches='tight')
    #plt.show()

    '''
    # Goodness of fit - use compensator to translate to poisson process for goodness of fit.
    #compensator
    uv_exp_compensator(T, mu,  alpha,  beta, smp)
    #goodness-of-fit
    interarrival_qq_exp(smp,'Simulation Probability Plot')
    interarrival_independence(smp)
    '''

    #uv_exp_forcast_nt(file_path,file_name, T, mu,  alpha,  beta, smp )
    return

def assess_players_mle():
    '''
    Assess MLE fit of univariate HP. Uses original and previously assessed data (xls).
    '''
        
    script_dir = os.path.dirname(__file__)

    #Players to estimate - based on MLE (read)
    player_mle_est = pd.read_csv(os.path.join(script_dir, 'mle_est_hp_3digits.csv'))
    player_mle_est = player_mle_est[['Player', 'mu','alpha','beta']]
    player_mle_est = player_mle_est.sort_values(['mu','alpha','beta'], ascending=[True,True, False])

    #injury data - (read)
    rel_path = 'D5_cols_to_use.xlsx' #'D5_processed_min_covar.xlsx'
    read_sheet = 'Sheet 1' #'player_position_data'
    with pd.ExcelFile(os.path.join(script_dir, rel_path)) as reader:
        sheet_injury = pd.read_excel(reader, sheet_name=read_sheet)
    #sort values
    sort_inj = sheet_injury.sort_values(['Player', 'Date'], ascending=[True, True])
    #nan
    sort_inj.replace(0.0, np.nan, inplace=True)

    #estimate
    #manual_lst = ['Player 16', 'Player 53', 'Player 64', 'Player 32', 'Player 40', 'Player 4', 'Player 6', 'Player 20', 'Player 24', 'Player 41', 'Player 3', 'Player 43', 'Player 2', 'Player 15', 'Player 34', 'Player 35', 'Player 12', 'Player 17', 'Player 29', 'Player 30', 'Player 54', 'Player 42']

    for player in player_mle_est['Player'].unique().tolist():
    #for player in manual_lst:
        print(player)

        #extract player data, sort/index and determing months duration of injury
        player_stat = sheet_injury.loc[sheet_injury['Player'] == player]
        player_stat_sort = player_stat.sort_values(by='Date')

        smp_inj = player_stat_sort.drop(player_stat_sort[player_stat_sort['Injury Type'] == 'NoInjury'].index)
        smp_inj.reset_index(drop=True, inplace=True)
        smp_inj['months'] = ((smp_inj['Date'] - smp_inj['Date'][0])/np.timedelta64(1, 'M'))
        smp_inj = smp_inj['months'].to_numpy()
        max_time = int(smp_inj[-1])+1 #32 months  

        #print(np.diff(smp_inj)) #tau's
        smp_inj = np.delete(smp_inj, 0) #removes first element
        smp = smp_inj
    
        mu_est,alpha_est,beta_est = [],[],[]

        parameters = _fit_grad_desc_mess(smp,max_time)

        mu_est.append(parameters[0])
        alpha_est.append(parameters[1])
        beta_est.append(parameters[2])
        iter_min = parameters[3]
        mess = parameters[4]
        iter_neg_ll = [ -x for x in parameters[5]]  
        #print(iter_neg_ll)

        title = player + ' MLE (mu:'+str(round(mu_est[0],4))+', alpha:'+str(round(alpha_est[0],4)) + \
                ', beta:'+str(round(beta_est[0],4))+')'
        filename = player + '_MLE' 

        #plot_fit(script_dir, player, smp,max_time,mu_est[0],alpha_est[0],beta_est[0],iter_neg_ll)
        plot_fit(script_dir,filename, title, smp,max_time,mu_est[0],alpha_est[0],beta_est[0],iter_neg_ll)

        #return
        '''
        #check if this is close to what was previously estimated.
        pre_est = est_parameters.loc[est_parameters['Player'] == player]
        threshold = 0.001
        para_dif = abs(mu_est - pre_est['mu'])+abs(alpha_est - pre_est['alpha'])+abs(beta_est - pre_est['beta'])
        print(player)
        if (float(para_dif) > threshold):
            print('retry estimate for: '+player)
            print('---estimated----')
            print(mu_est)
            print(alpha_est)
            print(beta_est)
            print('---already estimated----')
            print(pre_est['mu'])
            print(pre_est['alpha'])
            print( pre_est['beta'])      
        else:
            plot_fit(script_dir, player, smp,max_time,mu_est[0],alpha_est[0],beta_est[0],iter_neg_ll)'''
    return

def assess_players_misd():
    """`
    Implements Time scale separation and EM convergence of univariate Hawkes process
    with exponential decay (Lewis & Mohler, 2011). Stores players estimates in csv file.

    :param non
    :return: saves csv file of estimates
    """
    script_dir = os.path.dirname(__file__)

    #Players to estimate - based on MLE (read)
    player_mle_est = pd.read_csv(os.path.join(script_dir, 'mle_est_hp_3digits.csv'))
    player_mle_est = player_mle_est[['Player', 'mu','alpha','beta']]
    player_mle_est = player_mle_est.sort_values(['mu','alpha','beta'], ascending=[True,True, False])

    #injury data - (read)
    rel_path = 'D5_cols_to_use.xlsx' #'D5_processed_min_covar.xlsx'
    read_sheet = 'Sheet 1' #'player_position_data'
    with pd.ExcelFile(os.path.join(script_dir, rel_path)) as reader:
        sheet_injury = pd.read_excel(reader, sheet_name=read_sheet)
    #sort values
    sort_inj = sheet_injury.sort_values(['Player', 'Date'], ascending=[True, True])
    #nan
    sort_inj.replace(0.0, np.nan, inplace=True)

    #player_lst = ['Player 16', 'Player 53', 'Player 64', 'Player 32', 'Player 40', 'Player 4', 'Player 6', 'Player 20', 'Player 24', 'Player 41', 'Player 3', 'Player 43', 'Player 2', 'Player 15', 'Player 34', 'Player 35', 'Player 12', 'Player 17', 'Player 29', 'Player 30', 'Player 54', 'Player 42']
    #player_manual = ['Player 53']
    
    for player in player_mle_est['Player'].unique().tolist():
    #for player in player_manual:
        print(player)

        #extract player data, sort/index and determing months duration of injury
        player_stat = sheet_injury.loc[sheet_injury['Player'] == player]
        player_stat_sort = player_stat.sort_values(by='Date')

        smp_inj = player_stat_sort.drop(player_stat_sort[player_stat_sort['Injury Type'] == 'NoInjury'].index)
        smp_inj.reset_index(drop=True, inplace=True)
        smp_inj['months'] = ((smp_inj['Date'] - smp_inj['Date'][0])/np.timedelta64(1, 'M'))
        smp_inj = smp_inj['months'].to_numpy()
        max_time = int(smp_inj[-1])+1 #32 months  

        #print(np.diff(smp_inj)) #tau's
        smp_inj = np.delete(smp_inj, 0) #removes first element
        smp = smp_inj
    
        '''#try a number of times.
        works = 0
        for i in range(10):
            try:
                print('enters')
                print(i)
                mu_lst,alpha_lst,beta_lst,iter_neg_ll = uv_misd_em_fit(smp_inj,max_time,6)
                works = 1
                continue
            except Exception as e:
                print(player +' - convergence issue')
        if works == 0: #no valid estimate
            continue'''
            
        #try once
        mu_lst,alpha_lst,beta_lst,iter_neg_ll = uv_misd_em_fit(smp_inj,max_time,6)

        #only used last estimate value
        mu_est,alpha_est,beta_est = [],[],[]
        mu_est.append(mu_lst[-1])
        alpha_est.append(alpha_lst[-1])
        beta_est.append(beta_lst[-1])   
        print('-----')
        #print(iter_neg_ll)

        title = player + ' MISD (mu:'+str(round(mu_est[0],4))+', alpha:'+str(round(alpha_est[0],4)) + \
                ', beta:'+str(round(beta_est[0],4))+')'
        filename = player + '_MISD' 

        #plot_fit(script_dir, player, smp,max_time,mu_est[0],alpha_est[0],beta_est[0],iter_neg_ll)
        plot_fit(script_dir,filename, title, smp,max_time,mu_est[0],alpha_est[0],beta_est[0],iter_neg_ll)

    return

def main():



    #rank mu,alpha,beta
    #plot_parameters()

    #MLE
    #intensity,likelihood,goodness fit - of already parameters by MLE
    assess_players_mle()

    #MISD
    #assess_players_misd()
    return



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

def uv_exp_ll_grad(t,  mu,  alpha,  theta,  T):
    """
    Calculate the gradient of the likelihood function w.r.t. parameters mu, alpha, theta

    :param t: Bounded finite sample of the process up to time T. 1-d ndarray of timestamps
    :param mu: the exogenous intensity
    :param alpha: the infectivity factor alpha
    :param theta: intensity parameter of the delay density
    :param T: the maximum time
    :return: the gradient as a numpy.array of shape (3,). Gradients w.r.t. mu, alpha, theta respectively
    """

    phi, nphi = 0, 0
    Calpha,Ctheta = 0,0
    nmu , nalpha , ntheta = 0.,0.,0.
    N = len(t)
    j = 0
    d,r = 0., 0.

    nmu = 1/ mu
    Calpha = 1 - math.exp(-theta * (T - t[0]))
    Ctheta = alpha * (T - t[0]) * math.exp(-theta * (T - t[0]))

    for j in range(N-1):
        d = t[j+1] - t[j]
        r = T - t[j+1]

        ed = math.exp(-theta * d)
        #F = 1 - math.exp(-theta * r)
        try:
            F = 1 - math.exp(-theta * r)
        except: # number is too large
            F = 1 - math.exp(50)

        nphi = ed * (d * (1 + phi) + nphi)
        phi = ed * (1 + phi)
        lda = mu + alpha * theta * phi

        nmu = nmu + 1. / lda
        nalpha = nalpha + theta * phi / lda
        ntheta = ntheta + alpha * (phi - theta * nphi) / lda

        Calpha = Calpha + F
        Ctheta = Ctheta + alpha * r * (1 - F)

    return np.array([nmu - T, nalpha - Calpha, ntheta - Ctheta])



def uv_misd_em_fit(points_hawkes,T,max_time_phi,num_iter=500,reltol=1e-5):
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
        R_j = [] #expected total number of injuries that cause subsequent injury
        sum_ij = 0

        # E Step, update P 
        mu_ti = lamda_mu
        alpha_ti = lamda_alpha
        beta_ti = lambda_beta
        for j in range(N):
        
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
                    #print('--------')
                    #print(P[i][j])
                    sum_ij += P[i][j]
                    #print(sum_ij)
                else:
                    #kernel = (alpha_ti * beta_ti * np.exp(-beta_ti *tij))
                    P[i][i] = mu_ti/(kernel + mu_ti)
                    #print(i)
                    #print('++++++++++')
                    R_j.append(sum_ij/i)
                    sum_ij = 0
                    #R_j.append(sum_ij) #sum(Pij_flat)) #dynamic reproduction number R(t)
                    #sum_ij = 0
                    #R_j.append(sum(Pij_flat)/i) 

                
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
        print('mu: '+str(lamda_mu)+ ', alpha: '+str(lamda_alpha)+', beta: '+str(lambda_beta))
        # all elements should equal 1
        sum_rows =[]
        for i in range(N):
            sum_rows.append(sum(P[i]))

    #branching structure
    B = np.zeros_like(P)
    B[np.arange(len(P)), P.argmax(1)] = 1
    #check
    sum_dia_B = sum(np.diag(B))
    #sum_cros_dia_b = sum(B) - sum_dia_B


    
    print(R_j)
    print('----------')
    print(points_hawkes)
    print('----------')
    print(P)
    print('==========')
    print(B)
    print('==========')
    print(sum_dia_B)

    return mu_list,alpha_list,beta_list,like_list


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




def _fit_grad_desc_mess(t, T=None):
    """
    Given a bounded finite realization on [0, T], fit parameters with line search (L-BFGS-B).

    :param t: Bounded finite sample of the process up to time T. 1-d ndarray of timestamps. must be
    sorted (asc). dtype must be float.
    :param T: (optional) maximum time. If None, the last occurrence time will be taken.

    :return: the optimization result
    :rtype: scipy.optimize.optimize.OptimizeResult
    """

    N = len(t)

    ress = []

    # due to a convergence problem, we reiterate until the unconditional mean starts making sense
    for epoch in range(5):
        # estimate starting mu via the unconditional sample formula assuming
        # $\alpha \approx 0.2$
        mu0 = N * 0.8 / T

        # initialize other parameters randomly
        a0, th0 = np.random.rand(2)
        # todo: initialize th0 better ?
        myfactr = 1e2
        cb = CallbackFunctor(lambda x: -uv_exp_ll(t, x[0], x[1], x[2], T)) #save iterations
        minres = minimize(lambda x: -uv_exp_ll(t, x[0], x[1], x[2], T),
                        x0=np.array([mu0, a0, th0]),
                        jac=lambda x: -uv_exp_ll_grad(t, x[0], x[1], x[2], T),
                        bounds=[(1e-10, None), (1e-10, 1), (1e-10, None)],
                        callback=cb,
                        #method="Nelder-Mead",  options={"disp": False,"gtol": 1e-8})
                        #method="Newton-CG",  options={"disp": False, "ftol": myfactr * np.finfo(float).eps, "gtol": 1e-8})
                        method="L-BFGS-B", options={"disp": False, "ftol": myfactr * np.finfo(float).eps, "gtol": 1e-8})
        #Debug
        #print(cb.best_sols)       # contains all your collected solutions
        #print(cb.best_fun_vals)   # contains the corresponding objective function values

        #cb.save_sols("dummy.txt") # writes all solutions to a file 'dummy.txt'      

        ress.append(minres)
        mu, a, _ = minres.x
        mess = minres.message
        iter_min = minres.nit

        # take the unconditional mean and see if it makes sense
        #Napprox = mu * T / (1 - a)   
        Napprox = mu * T / (1.5 - a)   
        
        if abs(Napprox - N)/N < .01:  # if the approximation error is in range, break
            break
    
    #lik =  -uv_exp_ll(t, mu, a, _, T)
    return mu, a, _, iter_min,mess,cb.best_fun_vals[:-1],cb.best_sols[:-1]#obj_fun_lst # remove this
    #return mu, a, _, iter_min,lik,mess 


class CallbackFunctor:
    def __init__(self, obj_fun):
        self.best_fun_vals = [np.inf]
        self.best_sols = []
        self.num_calls = 0
        self.obj_fun = obj_fun
    
    def __call__(self, x):
        fun_val = self.obj_fun(x)
        self.num_calls += 1
        if fun_val < self.best_fun_vals[-1]:
            self.best_sols.append(x)
            self.best_fun_vals.append(fun_val)
   
    def save_sols(self, filename):
        sols = np.array([sol for sol in self.best_sols])
        np.savetxt(filename, sols)

# Control
if __name__=="__main__": 
    main() 
    