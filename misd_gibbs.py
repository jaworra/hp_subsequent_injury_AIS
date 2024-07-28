#from re import I
import random 
import math
from re import I
import numpy as np

import scipy.stats as stats
from scipy.stats import gamma,expon
from scipy.stats import uniform
from scipy.special import expit
from numpy.polynomial import legendre

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec 

import seaborn as sns # for density plots func, diagnostic_plt_2_chains

import os
import csv
from itertools import zip_longest #save latent files of different length

import time

#import corner #for parameter spread func,  diagnostic_plt_corner


def event_difference_list(events):
    """
    Evaluates lower triangluar matrix, represting time difference between events

    :param event: Bounded finite sample of the process up to time T. 1-d ndarray of timestamps
    :return: 2-d ndarray, lower triangluar matrix represting time difference between events
    """
    N = len(events)
    P = np.zeros((N,N))
    for j in range(N):
        for i in range(j,N):  # lower triangular matrix
            if i == 0: continue
            tij = events[i] - events[j] 
            P[i][j] = tij
    return P
 

def sample_parents(events, mu, alpha, beta, eventdifferencelist):
    """
    Stochastic declusting algorithm.  Calculates the background and triggering probabilityes and
    then class events by parent and child relationship.

    :param event: Bounded finite sample of the process up to time T. 1-d ndarray of timestamps
    :param mu: Background rate, float 
    :param alpha: Excitation constant, float
    :param beta: Decay rate, float
    :param eventdifferencelist: 2-d ndarray, lower triangluar matrix represting time difference between events
    :return: 3 lists, (child - offspring, parents - immagrants, shift_lst - child's time from parent)
    """

    N = len(events)
    P = np.zeros((N,N))
    for j in range(N):
        sum_ij=0
        for i in range(j,N):  # lower triangular matrix
            if i == 0: continue 
            tij = events[i] - events[j] 
        
            kernel = 0
            for k in range(0,i): #calculate kernel   #for k in range(0,i-1):
                tik = events[i] - events[k] 
                #kernel += (alpha * beta * np.exp(-beta * tik))      
                kernel += (alpha * np.exp(-beta * tik))     

            if i != j: 
                #P[i][j] = (alpha * beta * np.exp(-beta * tij))/(kernel + mu)
                P[i][j] = (alpha * np.exp(-beta * tij))/(kernel + mu) #this one
                #P[i][j] = (alpha * np.exp(-beta * tij))/kernel
            else: # Background activity
                P[i][i] = mu/(kernel + mu) #this one
                #P[i][i] = mu/kernel #does not sum to 1
            
    

    '''#deterministic - parent/child categorisation (with their event time)
    #branching structure - lower triangle matrix prob struct to 1/0 struct
    B = np.zeros_like(P)
    B[np.arange(len(P)), P.argmax(1)] = 1
    diag_B = np.diag(B) #check child

    child=[]
    parents=[]
    for k in range(N):
        if diag_B[k] == 1:
            parents.append(events[k]) 
        else:
            child.append(events[k])

    # Time difference from each child/parent
    shift_lst = []
    for idx, x in enumerate(np.argmax(B, axis=1)):
        if idx != x: #diag elements not included
            shift_lst.append(eventdifferencelist[idx][x])
    
    return B, child, parents , shift_lst'''

    #probablistic - parent/child categorisation (with their event time)
    child = []
    parents = []
    shift_lst = []
    parents.append(events[0]) #always parent

    for k in range(1,N):
        rw = P[k,] #row selection
        cat = np.random.choice(len(rw),size=1,replace=False, p=rw).item() #Is it parent/child, based on probabilities as weights
        if k==cat:
            parents.append(events[k])
        else:
            child.append(events[k])
            shift_lst.append(eventdifferencelist[k][cat])

    return child, parents, shift_lst
    

def gibbs_sampler(events, max_T, its = 100, return_parents = False, prior_a=0.01, prior_b = 0.01 ):
    """
    Fits a univariate Hawkes process with exponential decay function using the Stochastic delclustering Gibbs 
    algorithm. The algorithm exploits the conditional relationship, and sequentially samples parameters that are 
    weakly dependent, given latent variable B.

    :param event: Bounded finite sample of the process up to time T. 1-d ndarray of timestamps
    :param max_T: the maximum time
    :param its: int, maximum number of Gibbs iterations
    :param return_parents: boolean, return set of exogenous timestamps(immigrants)
    :param prior_a, prior_b: float, Gamma priors defaults alpha and beta = 0.01
    :return: lists, (mu, alpha, beta and parent class)
    :source: Poisson conjugate https://www.youtube.com/watch?v=lNrpPNk6InU (9.18m)
    """
    max_T_difference = max_T - events
    event_deltas = event_difference_list(events)


    mu_samp = np.random.rand()    #0.32255661746713105 
    alpha_samp = np.random.rand() #0.31135591711436705 
    beta_samp = np.random.rand()  #0.7226608044476059 

    #print('ini')
    #child, parent, shiftTimes = sample_parents(events, mu_samp, alpha_samp, beta_samp, event_deltas)
    child, parent, shiftTimes = sample_parents2(events, mu_samp, alpha_samp, beta_samp, event_deltas)

    #print(parent)
    #print(len(parent))
    #print(len(child))
    #return

    mu_l,alpha_l,beta_l = [],[],[]
    S0_l,Sj_l = [],[]
    for i in range(its):

        #print('iter : '+str(i) + ' ,parent: '+str(len(parent))+ ' ,child: '+str(len(child)))
        mu_samp  = gamma.rvs(prior_a + len(parent), scale= 1/(prior_b+1)) / max_T
        mu_l.append(mu_samp)

        H_tilde =  sum(expon.cdf(x=max_T_difference, scale=1/alpha_samp))

        alpha_samp = gamma.rvs(prior_a+ len(child), scale= 1/(prior_b + H_tilde))
        alpha_l.append(alpha_samp)

        beta_samp = gamma.rvs(prior_a+len(shiftTimes), scale= 1/(prior_b + sum(shiftTimes)))
        beta_l.append(beta_samp)

        #child, parent, shiftTimes = sample_parents(events, mu_samp, alpha_samp, beta_samp, event_deltas)
        child, parent, shiftTimes = sample_parents2(events, mu_samp, alpha_samp, beta_samp, event_deltas)

        S0_l.append(len(parent))
        Sj_l.append(len(child))
        
        avg_mu = sum(mu_l)/len(mu_l)
        avg_alpha = sum(alpha_l)/len(alpha_l)
        avg_beta = sum(beta_l)/len(beta_l)
        #print('relaxed coef mu: '+str(avg_mu)+ ' ,alpha: '+str(avg_alpha)+ ' ,beta: '+str(avg_beta))

    if return_parents:
        return mu_l, alpha_l, beta_l , S0_l, parent, child
    return mu_l, alpha_l, beta_l , S0_l



def sample_parents2(events, mu, alpha, beta, eventdifferencelist):
    """
    re-written Stochastic declustering algorithm.  Calculates the background and triggering probabilities and
    then class events by parent and child relationship.

    :param event: Bounded finite sample of the process up to time T. 1-d ndarray of timestamps
    :param mu: Background rate, float 
    :param alpha: Excitation constant, float
    :param beta: Decay rate, float
    :param eventdifferencelist: 2-d ndarray, lower triangular matrix representing time difference between events
    :return: 3 lists, (child - offspring, parents - immigrants, shift_lst - child's time from parent)
    """

    parents = []
    child =  []
    shift_lst = []
    n = len(eventdifferencelist)
    for (i, eventdiff) in enumerate(eventdifferencelist):
        eventdiff = list(filter(lambda num: num != 0, eventdiff))
        parent = event_parent_prob2(events[i], mu, alpha, beta, eventdiff)

        if parent == 0: #caused from background
            parents.append(events[i])
        else:
            child.append(events[i])
            shift_lst.append(events[i] - events[parent - 1]) #double check this. [parent - 1] !!!. List position 

    return child, parents, shift_lst
    

def event_parent_prob2(event, bg, kappa, kern, eventdifference):
    #proablity its a child event or, been caused by another.
    n = len(eventdifference)

    #checks
    #true_probs = []
    #kernel = 0

    probs = []
    probs.append(bg) #first row P(of mu/background - S_0 )
    for i in range(1,n+1):
        
        #probs_temp = kappa * np.exp(-kern * eventdifference[i-1])
        #probs.append(probs_temp)
        probs_temp = kappa * kern * np.exp(-kern * eventdifference[i-1]) #P(child event)
        probs.append(probs_temp) 

        #checks below
        #kernel += kappa * kern * np.exp(-kern * eventdifference[i-1])
        #true_probs.append(probs_temp/(bg+kernel))
    #if n>1:
        #true_probs.append(bg/(bg+kernel))
        #sum_t0_1 = np.sum(true_probs) #check


    #previous algorithm - uses denominator
    #P[i][j] = (alpha * np.exp(-beta * tij))/(kernel + mu) #probability j is triggered by another event.
    #P[i][i] = mu/(kernel + mu) # probability j is a background event

    n_probs = probs / np.sum(probs) #normalising - cheat (for rounding) so probability density sums to 1.
    cat_prob = np.random.choice(len(n_probs),size=1,replace=False, p=n_probs).item()  #https://stackoverflow.com/questions/46539431/np-random-choice-probabilities-do-not-sum-to-1
   
   
    return cat_prob 


#--------------------------------------------------------------------
#Save/Load for Diagnostics.

def save_chain(mu,alpha,beta,S0,file_name = ''):
    """
    Saves MCMC sampled parameters to directory for diagnostic etc. Strick arguments requirements.

    :param mu: Background rate, list of float 
    :param alpha: Excitation constant, list of float 
    :param beta: Decay rate, list of float 
    :param S0: Number of background events, list of integers 
    :param file_name: file name (i.e test.csv), if not given a default is used based on time.
    :return: none
    """


    timestr = time.strftime("%d%m%y_%H%M")
    #default filename if not provided
    if not file_name:
        file_name = str(timestr)+ '_'+str(len(mu))+'_chain.csv'

    #sets default dir
    script_dir = os.path.dirname(__file__)
    parent_dir = os.path.abspath(script_dir + "/../")
    directory = parent_dir + '/data/mcmc_chains'
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    #duplicate filename exist append timestamp
    if os.path.isfile(directory+'/'+file_name):
        file_name = file_name.replace('.csv','')
        file_name = file_name + '_' + str(timestr)+ '.csv'

    #saving..
    rows = zip(mu,alpha,beta,S0)
    with open(directory+'/'+file_name, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['mu','alpha','beta','so'])
        for row in rows:
                writer.writerow(row)
        csv_file.close()

    print('MCMC file saved: '+ file_name)
    return

def load_chain(file_name, dir_name=''):
    """
    Loads MCMC parameters samples.

    :param file_name: filename to load, string.
    :param dir_name: location of directory, string. If not provide looks in default.
    :param file_name: file name (i.e test.csv), if not given a default is used based on time.
    :returns: 4 list (mu, alpha,beta as floats and s0 as integer)  
    """

    if not dir_name:
        script_dir = os.path.dirname(__file__)
        parent_dir = os.path.abspath(script_dir + "/../")
        directory = parent_dir + '/data/mcmc_chains'

    mu_l,alpha_l,beta_l,s0_l = [],[],[],[]
    with open(directory+'/'+file_name, mode='r') as csv_file:
        #skip header
        next(csv_file)
        for line in csv_file.readlines():
            mu,alpha,beta,s0 = line.strip().split(',')
            mu_l.append(float(mu))
            alpha_l.append(float(alpha))
            beta_l.append(float(beta))
            s0_l.append(int(s0))

    return mu_l,alpha_l,beta_l,s0_l


def save_latent_variables(parent,child,iter,file_name = ''):
    """
    Saves parent/child event classification. Strick arguments requirements.

    :param parent: Immagrant event, list of float 
    :param child: Parent, list of float 
    :param file_name: file name (i.e test.csv), if not given a default is used based on time.
    :return: none
    """


    timestr = time.strftime("%d%m%y_%H%M")
    #default filename if not provided
    if not file_name:
        file_name = str(timestr)+ '_'+str(iter)+'_latent_variable.csv'

    #sets default dir
    script_dir = os.path.dirname(__file__)
    parent_dir = os.path.abspath(script_dir + "/../")
    directory = parent_dir + '/data/mcmc_chains'
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    #duplicate filename exist append timestamp
    if os.path.isfile(directory+'/'+file_name):
        file_name = file_name.replace('.csv','')
        file_name = file_name + '_' + str(timestr)+ '.csv'

    #saving..
    rows = [parent,child]
    with open(directory+'/'+file_name, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['parent','child'])
        for row in zip_longest(*rows):
            writer.writerow(row)
        csv_file.close()

    print('MCMC latent variable file saved')
    return 

def load_latent_variable(file_name, dir_name=''):
    """
    Loads latent parameters samples.

    :param file_name: filename to load, string.
    :param dir_name: location of directory, string. If not provide looks in default.
    :param file_name: file name (i.e test.csv), if not given a default is used based on time.
    :returns: 4 list (parent, child as float)  
    """

    if not dir_name:
        script_dir = os.path.dirname(__file__)
        parent_dir = os.path.abspath(script_dir + "/../")
        directory = parent_dir + '/data/mcmc_chains'


    parent_l,child_l = [],[]
    with open(directory+'/'+file_name, mode='r') as csv_file:
        #skip header
        next(csv_file)
        for line in csv_file.readlines():
            parent,child = line.strip().split(',')
            parent_l.append(parent)
            child_l.append(child)
    
    parent_l = [float(x) for x in parent_l if x != '']
    child_l = [float(x) for x in child_l if x != '']

    return parent_l,child_l

#--------------------------------------------------------------------
#Diagnostics for Gibbs sampler.


def acorr(x, ax=None):
    #modified from - https://stackoverflow.com/questions/27541290/bug-of-autocorrelation-plot-in-matplotlib-s-plt-acorr
    x = np.array(x, dtype=np.float32)
    if ax is None:
        ax = plt.gca()
    x = x - x.mean()

    autocorr = np.correlate(x, x, mode='full')
    autocorr = autocorr[x.size:]
    autocorr /= autocorr.max()

    return ax.stem(autocorr, markerfmt=' ', use_line_collection=True, linefmt='grey') #


def diagnostic_plt(mu,alpha,beta,S0, save_file = False):
    line_col = 'dimgrey'
    fig_x = 16
    fig_y = 10

    mu_mu = np.mean(mu)
    alpha_mu = np.mean(alpha)
    beta_mu = np.mean(beta)
    s0_mu = np.mean(S0)

    #manual bin selection
    bin_width = 50
    f = plt.figure(figsize=(fig_x,fig_y))
    grid = plt.GridSpec(3, 3)  # 2 rows 3 cols

    ax1 = f.add_subplot(grid[0, :2])
    ax1.plot(mu, color = line_col) 
    ax1.set_ylabel('Sample value') 
    ax1.set_title(r'Trace of $\mu$')

    ax2 = f.add_subplot(grid[1, :2])
    ax2.plot(alpha, color = line_col) 
    ax2.set_ylabel('Sample value') 
    ax2.set_title(r'Trace of $\alpha$')

    ax3 = f.add_subplot(grid[2, :2])
    ax3.plot(beta, color = line_col) 
    ax3.set_ylabel('Sample value') 
    ax3.set_title(r'Trace of $\beta$')

    #histograms
    ax4 = f.add_subplot(grid[0, 2])
    y1,x1,_ = ax4.hist(mu, bins=bin_width, color = line_col )
    ax4.vlines(mu_mu,ymin=0, ymax=y1.max(), colors='r')
    ax4.set_ylabel('Frequency') 
    ax4.set_title('Mu Samples')

    ax5 = f.add_subplot(grid[1, 2:])
    y1,x1,_  = ax5.hist(alpha, bins=bin_width, color = line_col )
    ax5.vlines(alpha_mu,ymin=0, ymax=y1.max(), colors='r')
    ax5.set_ylabel('Frequency') 
    ax5.set_title('Alpha Samples')

    ax6 = f.add_subplot(grid[2, 2:])
    y1,x1,_  = ax6.hist(beta, bins=bin_width, color = line_col )
    ax6.vlines(beta_mu,ymin=0, ymax=y1.max(), colors='r')
    ax6.set_title('Beta Samples')
    plt.show()

    #so
    f = plt.figure(figsize=(fig_x,5))
    grid = plt.GridSpec(1, 3)  # 2 rows 3 cols

    ax1 = f.add_subplot(grid[0, :2])
    ax1.plot(S0, color = line_col) 
    ax1.set_ylabel('Sample value') 
    ax1.set_title(r'Trace of S0')

    ax2 = f.add_subplot(grid[0, 2])
    y1,x1,_ = ax2.hist(S0, bins=bin_width, color = line_col )
    ax2.vlines(s0_mu,ymin=0, ymax=y1.max(), colors='r')
    ax2.set_ylabel('Frequency') 
    ax2.set_title('S0 Samples')


    #autocorrelation - new
    #set lag
    lag_size = 20 
    f = plt.figure(figsize=(fig_x,fig_y))
    grid = plt.GridSpec(3, 3)  # 2 rows 3 cols

    ax1 = f.add_subplot(grid[0, :2])
    acorr(mu, ax=ax1)
    ax1.set_ylabel('correlation')
    ax1.set_title('Mu, ACF full length')

    ax2 = f.add_subplot(grid[1, :2])
    acorr(alpha, ax=ax2)
    ax2.set_ylabel('correlation')
    ax2.set_title('Alpha, ACF full length')

    ax3 = f.add_subplot(grid[2, :2])
    acorr(beta, ax=ax3)
    ax3.set_ylabel('correlation')
    ax3.set_title('Beta, ACF full length')

    #lag 
    ax4 = f.add_subplot(grid[0, 2])
    acorr(mu, ax=ax4)
    #ax4.set_ylabel('correlation')
    ax4.set_title('Mu, lag '+ str(lag_size))
    ax4.set_xlim([0, lag_size])

    ax5 = f.add_subplot(grid[1, 2:])
    acorr(alpha, ax=ax5)
    #ax5.set_ylabel('correlation')
    ax5.set_title('Alpha, lag '+ str(lag_size))    
    ax5.set_xlim([0, lag_size])

    ax6 = f.add_subplot(grid[2, 2:])
    acorr(beta, ax=ax6)
    #ax6.set_ylabel('correlation')
    ax6.set_title('Beta, lag '+ str(lag_size))
    ax6.set_xlim([0, lag_size])

    plt.show()
    return
    

def diagnostic_plt_corner(mu,alpha,beta,S0, save_file = False):
    '''Highlights parameter space'''

    labels = ['Mu','Alpha','Beta', 'S0']
    samples =  np.array([mu,alpha,beta,S0])
    samples = np.transpose(samples)

    figure = corner.corner(samples, show_titles=True,labels=labels,lot_datapoints=True,quantiles=[0.16, 0.5, 0.84])
    plt.figure
    plt.show()
    return


def diagnostic_plt_2_chains(mu,alpha,beta,mu2,alpha2,beta2, withfreq = 1):
    """
    In reviewing MCMC convergence, plots two chains for comparison.
    param mu,alpha,beta and param  mu2,alpha2,beta2: for exponential decay paramters
    param withfreq: integer, defaults for 1 for density else just trace plots,
    """
    fig_x = 16
    fig_y = 10
    line_col = 'dimgrey'
    line_col_2 = 'r'

    #NOTE: Kernel density (seaborn) uses scott estimate for bandwidth selection.
    #Leading to high Y values, in some cases.
    #https://stats.stackexchange.com/questions/90656/kernel-bandwidth-scotts-vs-silvermans-rules
    #TODO: 1.Implement own bandwidth selection here.
    # 2. Autocorrelation comparisons.
    if withfreq == 1:
        line_w = 2

        f = plt.figure(figsize=(fig_x,fig_y))
        grid = plt.GridSpec(3, 3)  # 2 rows 3 cols

        ax1 = f.add_subplot(grid[0, :2])
        ax1.plot(mu, color = line_col)   # density=False would make counts
        ax1.plot(mu2, color = line_col_2)  
        ax1.set_ylabel('Sample value') 
        ax1.set_title(r'Trace of $\mu$')

        ax2 = f.add_subplot(grid[1, :2])
        ax2.plot(alpha,color = line_col)   # density=False would make counts
        ax2.plot(alpha2, color = line_col_2)   # density=False would make count
        ax2.set_ylabel('Sample value') 
        ax2.set_title(r'Trace of $\alpha$')

        ax3 = f.add_subplot(grid[2, :2])
        ax3.plot(beta, color = line_col)   # density=False would make counts
        ax3.plot(beta2, color = line_col_2)  
        ax3.set_ylabel('Sample value') 
        ax3.set_title(r'Trace of $\beta$')


        ax4 = f.add_subplot(grid[0, 2])
        sns.distplot(mu, hist = False, kde = True,
                    kde_kws = {'linewidth': line_w},
                    ax=ax4,color = line_col) 
        sns.distplot(mu2, hist = False, kde = True,
                    kde_kws = {'linewidth': line_w},
                    ax=ax4,color = line_col_2) 
        ax4.set_ylabel('Density') 
        ax4.set_title(r'Sample density of $\mu$')

        ax5 = f.add_subplot(grid[1, 2:])
        sns.distplot(alpha, hist = False, kde = True,
                    kde_kws = {'linewidth': line_w},
                    ax=ax5,color = line_col) 
        sns.distplot(alpha2, hist = False, kde = True,
                    kde_kws = {'linewidth': line_w},
                    ax=ax5,color = line_col_2) 
        ax5.set_ylabel('Density') 
        ax5.set_title(r'Sample density of $\alpha$')

        ax6 = f.add_subplot(grid[2, 2:])
        sns.distplot(beta, hist = False, kde = True,
                    kde_kws = {'linewidth': line_w},
                    ax=ax6,color = line_col) 
        sns.distplot(beta2, hist = False, kde = True,
                    kde_kws = {'linewidth': line_w},
                    ax=ax6,color = line_col_2) 
        ax6.set_ylabel('Density') 
        ax6.set_title(r'Sample density of $\beta$')

    else:
        #trace plot with histogram *fix this plot should 
        f = plt.figure(figsize=(fig_x,fig_y))

        ax1 = f.add_subplot(3,1,1)
        ax1.plot(mu)  # density=False would make counts
        ax1.plot(mu2)  
        ax1.set_ylabel('Sample value') 
        ax1.set_title(r'Trace of $\mu$')

        ax2 = f.add_subplot(3,1,2)
        ax2.plot(alpha)  # density=False would make counts
        ax2.plot(alpha2)  # density=False would make count
        ax2.set_ylabel('Sample value') 
        ax2.set_title(r'Trace of $\alpha$')

        ax3 = f.add_subplot(3,1,3)
        ax3.plot(beta)  # density=False would make counts
        ax3.plot(beta2) 
        ax3.set_ylabel('Sample value') 
        ax3.set_title(r'Trace of $\beta$')
        
    plt.show()   

    return


#--------------------------------------------------------------------
#Metrics below. Likelihood first.


def report_est_para_like(mu,alpha,beta,S0, events, T):
   
    mu_mu = np.mean(mu)
    mu_std = np.std(mu)
    alpha_mu = np.mean(alpha)
    alpha_std = np.std(alpha)
    beta_mu = np.mean(beta)
    beta_std = np.std(beta)

    s0_mu = np.mean(S0)

    #terminal output statistics
    print('-------------chain stats--------------')
    print('Avg: Mu: '+ str(round(mu_mu,4)) + '| Alpha: '+ str(round(alpha_mu,4)) +'| Beta: '+ str(round(beta_mu,4)) )
    print('Std: Mu: '+ str(round(mu_std,3)) + '| Alpha: '+ str(round(alpha_std,3)) +'| Beta: '+ str(round(beta_std,3)) )
    print('--------------------------------------')

    like = uv_exp_ll(events, mu_mu, alpha_mu, beta_mu, T)

    #print(str(round(mu_mu,4)) + ' ,'+ str(round(alpha_mu,4)) +' ,'+ str(round(beta_mu,4)) +' ,'+ str(round(like,4)) )
    print(str(len(mu))  + ' & '+  str(round(mu_mu,4)) + ' & '+ str(round(alpha_mu,4)) +' & '+ str(round(beta_mu,4)) +' & '+ str(round(like,4)) + ' & '+  str(len(events)) )

    return


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



    
#--------------------------------------------------------------------
#MCMC chain postprossses 

def burnin_thinning(mu, alpha, beta, burnin, thin=0):

    #burnin
    mu = mu[burnin:]
    alpha = alpha[burnin:]
    beta = beta[burnin:]

    if thin != 0:
        del mu[::thin]
        del alpha[::thin]
        del beta[::thin]


    return mu, alpha, beta

'''
#testing
print('testings starting')
smp = np.array([ 1.97129305,  2.23413212,  2.46411631,  3.15406887,  6.76810612,
        8.37799544,  8.54226986, 12.87911456, 13.40479271, 13.56906713,
       13.93047085, 14.12760016, 14.88326249, 15.34323087, 16.75599088,
       17.41308856, 17.70878252, 19.71293045, 19.94291464, 23.524097  ,
       27.53239286, 27.76237705, 28.18949054, 28.45232962, 30.42362266,
       31.01501058, 31.67210826, 32.09922175])
max_t =32 


mu_l, alpha_l, beta_l , S0_l = gibbs_sampler(smp, max_t, its = 100,)
print('completed')

#output
#parent
[1.97129305, 2.23413212, 2.46411631, 3.15406887, 6.76810612, 8.37799544, 8.54226986, 12.87911456, 13.40479271, 13.56906713,
13.93047085, 14.12760016, 14.88326249, 15.34323087, 17.70878252, 23.524097, 27.53239286, 27.76237705, 28.18949054, 30.42362266,
 31.01501058, 32.09922175]
#len(parent)
22

'''
