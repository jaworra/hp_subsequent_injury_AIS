from pickle import TRUE
import numpy as np
import abc 
import random 
import math
from matplotlib import pyplot as plt

#for the max log likelihood
from scipy.optimize import minimize

import scipy
import statsmodels.api as sm #for qq plot
import seaborn as sns


import sys
sys.path.insert(0, 'weekly_updates/wk_63')
from uni_exp import _fit_grad_desc, uv_exp_sample_ogata, uv_exp_sample_branching

def interarrival_independence(smp):
    tau =  np.empty(len(smp), dtype=np.float64)
    t_pre = 0
    for i, t_i in enumerate(smp):
        tau[i] = t_i - t_pre
        t_pre = t_i

    plt.acorr(tau, maxlags = 20)
    plt.show()
    return


def interarrival_qq_exp(smp,plot_both = TRUE, size = [12,5], filename = 'QQ Plot'):
    """Takes event times and converts to inter-arrival and plots 
        exponential fit - qq plot/histogram"""

    tau =  np.empty(len(smp), dtype=np.float64)
    t_pre = 0
    for i, t_i in enumerate(smp):
        tau[i] = t_i - t_pre
        t_pre = t_i

    #tests.
    #tau = np.random.exponential(scale = 1, size=1000)

    #ref qqplot vs prob plot
    #https://tungmphung.com/qq-versus-pp-plot-versus-probability-plot/
    if plot_both:
        fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(size[0],size[1]))
        #scipy.stats.probplot(tau, dist="expon", plot=ax1) #probability plot

        sns.distplot(tau, hist=True, kde=False, 
             bins=int(len(tau)/5), color = 'grey', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4}, ax=ax1)
        ax1.set_title('Histogram inter-arrival')
        sm.qqplot(tau,line='45',fit=True,dist=scipy.stats.expon, markerfacecolor='darkgrey',
                  markeredgecolor='k', color='grey',ax=ax2)
        #sm.qqplot(tau,fit=True,dist=scipy.stats.expon, markerfacecolor='darkgrey',color='grey',ax=ax2)
        ax2.set_title('Exponential Q-Q Plot')

        #code block below from below. Todo: check and return QQ value not prob plot values.
        #https://stackoverflow.com/questions/36151265/bug-stats-probplot-returns-as-text-inside-the-plot-r-not-r2-as-it-says
        osm,osr = scipy.stats.probplot(tau, fit=True, plot=None, dist=scipy.stats.expon)
        x=osm[0];y=osm[1]
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
        #print(r_value)
        print(r_value**2)
    else:
        fig,ax= plt.subplots(1, 1,figsize=(size[0],size[1]))
        sm.qqplot(tau,line='45',fit=True,dist=scipy.stats.expon,markerfacecolor='darkgrey',
                  markeredgecolor='k', ax=ax)
        #plt.title('Q-Q Plot')
        #for probability plot uncomment below
        #scipy.stats.probplot(tau, dist="expon", plot=plt)

    plt.show()
    return



def uv_exp_compensator(max_time,  mu,  alpha,  beta,smp ):


    #smp = uv_exp_sample_branching(max_time, mu, alpha, beta)

    #generalise summation form
    lda_ar = [mu + np.sum(alpha  * np.exp(-beta * (x - smp[smp < x]))) \
            for x in np.arange(0, max_time, .1)]    
    #expectation         
    lda_ar_est = [mu + np.sum(alpha  * np.exp(-beta * (x - smp[smp < x]))) \
            for x in np.arange(0, max_time, .1)]
    
    #compensator
    # https://github.com/Pat-Laub/hawkesbook/blob/35f152f152ad68bdbcaf687e5c93c83046201c54/hawkesbook/hawkes.py
    # line 26
    # https://github.com/Pat-Laub/hawkesbook/blob/35f152f152ad68bdbcaf687e5c93c83046201c54/tests/basic.py
    # line 71
    #comp_est  = [mu*x + np.sum(alpha/beta * (1 - np.exp(-beta *( x - smp[smp < x])))) \
    #            for x in np.arange(0, max_time, .1)]
    comp_est  = [mu*max_time + np.sum(alpha/beta * (1 - np.exp(-beta *( x - smp[smp < x])))) \
                for x in np.arange(0, max_time, .1)]               

    # transformed N(t)
    #https://github.com/Pat-Laub/hawkesbook/blob/35f152f152ad68bdbcaf687e5c93c83046201c54/hawkesbook/hawkes.py#L44
    # lin 54
    Lam=0
    mu_as_pre = mu
    t_pre = 0
    comp_tran = np.empty(len(smp), dtype=np.float64)
    for i, t_i in enumerate(smp):
        Lam += mu * (t_i - t_pre) + (
                (mu_as_pre - mu)/beta *
                (1 - np.exp(-beta*(t_i - t_pre))))
        comp_tran[i] = Lam

        mu_as_pre = mu + (mu_as_pre - mu) * (np.exp(-beta * (t_i - t_pre))) + alpha
        t_pre = t_i

    

    # plot intensity and counting plots
    fig, axs = plt.subplots(2,2,figsize=(15,2))
    
    # N(t)
    ar_pp = sorted(smp) 
    axs[0,0].set_ylabel("$N(t)$")
    axs[0,0].set_xlabel('$t$')

    axs[0,0].plot(ar_pp, np.cumsum(np.ones_like(ar_pp)),drawstyle='steps-pre',label="N(t)")
    axs[0,0].plot([0,max_time],[0, np.average(np.cumsum(np.ones_like(sorted(smp)))/sorted(smp))*max_time] , c='red',linestyle='--',lw=1, label="$\mathbb{E}[N(t)]$")
    axs[0,0].set_title('')
    axs[0,0].legend(loc='upper left')
    xmin, xmax = axs[0,0].get_xlim() #force axis`

    # Conditional intensity function
    axs[0,1].set_ylabel("$\lambda^*(t)$")
    axs[0,1].set_xlabel("$t$")
    axs[0,1].plot(smp, np.ones_like(smp) * mu, 'k.',label="arrivals")
    _ = axs[0,1].plot(np.arange(0, max_time, .1), lda_ar, color= '#1f77b4',label="$\lambda^*(t)$")#'b-' ##ff7f0e orange
    axs[0,1].hlines(np.average(lda_ar),xmin=0, xmax=max_time,  colors='red',linestyle='--',lw=1, label="$\mathbb{E}[\lambda^*(t)]$")
    axs[0,1].set_xlim([xmin,xmax])
    axs[0,1].set_title('')
    axs[0,1].legend(loc='upper left')

    # Compensator  comp_tran
    axs[1,0].set_ylabel("$\Lambda_t$")
    axs[1,0].set_xlabel("$t$")
    _ = axs[1,0].plot(np.arange(0, max_time, .1), comp_est, color= 'violet',label="$\Lambda(t)$")#'b-' ##ff7f0e orange
    #_ = axs[1,0].plot(ar_pp, comp_tran, color= 'violet',label="$\Lambda(t)$")#'b-' ##ff7f0e orange
    axs[1,0].set_title('')
    axs[1,0].legend(loc='upper left')

    # Transformed process N*(t)
    #axs[1,1].set_ylabel("$N*(t)$")
    axs[1,1].set_xlabel("$t$")
    axs[1,1].plot(ar_pp, comp_tran,  c='#1f77b4', label="$N^*(t)$")
    #axs[1,1].plot([0,ar_pp[-1]],[0, np.average(np.cumsum(np.ones_like(sorted(comp_tran)))/sorted(comp_tran))*ar_pp[-1]] , c='red',linestyle='--',lw=1, label="$\mathbb{E}[N(t)]$")
    #axs[1,1].plot(comp_tran, np.cumsum(np.ones_like(ar_pp)) , c='red',linestyle='--',lw=1, label="$\mathbb{E}[N^*(t)]$")
    #axs[1,1].plot(np.arange(0, max_time, 1), comp_tran , c='red',linestyle='--',lw=1, label="$\mathbb{E}[N^*(t)]$")
    axs[1,1].set_title('')
    axs[1,1].legend(loc='upper left')


    print(len(smp))
    print(np.cumsum(np.ones_like(ar_pp)))

    plt.show()
    return