import numpy as np
import abc 
import random 
import math
from matplotlib import pyplot as plt
plt.style.use('ggplot')

#for the max log likelihood
from scipy.optimize import minimize

import scipy

import time
import os
import sys
sys.path.insert(0, 'weekly_updates/wk_63')
from uni_exp import _fit_grad_desc, uv_exp_sample_ogata, uv_exp_sample_branching


def disc_step_func(y, for_t = 12, inc=0.01):
    """ Discretises forecasting N(t) samples for use in forecasting (CI)
    Args:
        y (list): predicted times 
        for_t (int, optional): Domain max value. Defaults to 12.
        inc (float, optional): Discretised domain value. Defaults to 0.01.

    Returns:
        x_up,y_up (two list): both discretised function and domain.  
    """

    x_inc = 0
    cnt = 0
    y_up,x_up = [],[]
    
    for i in y:
        
        while x_inc < i:
            x_inc += inc
            x_up.append(x_inc)
            y_up.append(cnt)

        cnt += 1
        y_up.pop()
        y_up.append(cnt)
    
    while x_inc <  for_t:
        x_inc += inc
        x_up.append(x_inc)
        y_up.append(cnt)

    return x_up , y_up

def uv_exp_forcast_nt(max_time, mu, alpha, beta, smp, for_t = 12, sim_n =15,\
                     show_lines= False, ret_forecast = False, file_name = ''):
    """ Plots N(t) fit, conditional intensity fit and forecasts via monte carlo given
        estimate parameters.
    Args:
        max_time (float): time window for estimate
        mu (float): baseline
        alpha (float): excitation parameter
        beta (float): decay parameter
        smp (np.array): Times of arrivals
        for_t (int, optional): amount of time window. Defaults to 12 given in months for a year.
        sim_n (int, optional): number of simulations. Defaults to 15.
        show_lines (bool, optional): show each. Defaults to False.
        ret_forecast (bool, optional): return forecast, median arrival times (array). Defaults to False.
        file_name (str, optional): saves if variable is not empty (relative path). Defaults to empty.
    """
   
    line_col = 'dimgrey' # color= '#1f77b4' #for blue
    main_col = 'red'
    colors = plt.cm.gray(np.linspace(0.4,0.8,sim_n)) #forecasting lines
    fig_x = 15
    fig_y = 7

    #generalise summation form
    lda_ar = [mu + np.sum(alpha  * np.exp(-beta * (x - smp[smp < x]))) \
            for x in np.arange(0, max_time, .1)]    
    '''
    #expectation         
    lda_ar_est = [mu + np.sum(alpha  * np.exp(-beta * (x - smp[smp < x]))) \
            for x in np.arange(0, max_time, .1)]
    '''

    fig, axs = plt.subplots(3,1,figsize=(fig_x,fig_y))
    # N(t)
    ar_pp = sorted(smp) 
    axs[0].set_ylabel("$N(t)$")
    axs[0].set_xlabel('$t$')
    axs[0].plot([0,ar_pp[0]], [0,0],color = line_col) #start of step function
    axs[0].plot(ar_pp, np.cumsum(np.ones_like(ar_pp))-1,color = line_col, drawstyle='steps-pre',label="N(t)")
    axs[0].plot([0,smp[-1]],[0, len(smp)] , c=main_col,linestyle='--',lw=1, label="$\mathbb{E}[N(t)]$")
    
    #axs[0].set_title(file_name)
    axs[0].legend(loc='upper left')
    xmin, xmax = axs[0].get_xlim() #force axis`

    # Conditional intensity function
    axs[1].set_ylabel("$\lambda^*(t)$")
    axs[1].set_xlabel("$t$")
    axs[1].plot(smp, np.ones_like(smp) * mu, 'k.',label="arrivals") #,c='red'
    axs[1].plot(np.arange(0, max_time, .1), lda_ar, color = line_col ,label="$\lambda^*(t)$")#'b-' ##ff7f0e orange
    axs[1].set_xlim([xmin,xmax]) 
    axs[1].set_title('')
    axs[1].legend(loc='upper left')

    # forcast
    axs[2].set_ylabel("$N(t)$")
    axs[2].set_xlabel("$t$")


    for_nt, for_nt_dis = [],[] #matrix list of unequal size (Poisson distributed arrival times)
    for i in range(sim_n):
        for_smp = uv_exp_sample_branching(for_t, mu, alpha, beta )
        for_nt.append(for_smp.tolist())
        
        #discretise
        if len(for_smp) > 0: #predict on only valid forecasts
            x_dis,y_dis = disc_step_func(for_smp, for_t, 0.01)
            for_nt_dis.append(y_dis)
            if show_lines: #plot each line
                axs[2].plot(x_dis, y_dis,label="N(t)",color=colors[i]) #plot all colors
            

    #calculate discretise f(.) max val.
    for_nt_dis_ar = np.array(for_nt_dis, dtype=object)
    
    #stats prediction
    #https://docs.scipy.org/doc/numpy-1.9.1/reference/generated/numpy.percentile.html
    #note: defaults to linear interpolation. 
    #CI intervals, 95 and 68 precentile
    '''per_50_arr = list(np.percentile(for_nt_dis_ar, 50, axis=0))
    per_32_arr = list(np.percentile(for_nt_dis_ar, 32, axis=0))
    per_68_arr = list(np.percentile(for_nt_dis_ar, 68, axis=0))
    per_95_arr = list(np.percentile(for_nt_dis_ar, 95, axis=0))'''

    per_32_arr = list(np.percentile(for_nt_dis_ar, 32, interpolation='lower', axis=0))
    per_50_arr = list(np.percentile(for_nt_dis_ar, 50, interpolation='midpoint', axis=0))
    per_68_arr = list(np.percentile(for_nt_dis_ar, 68, interpolation='higher', axis=0))
    per_95_arr = list(np.percentile(for_nt_dis_ar, 95, interpolation='higher', axis=0))


    
    axs[2].plot(x_dis, per_50_arr, color = main_col, alpha = 1, label="median")  
    axs[2].fill_between(x_dis, per_68_arr, per_32_arr, color = 'black',\
                        alpha = 0.4, label="68% CI")
    
    #maximum
    axs[2].fill_between(x_dis, per_95_arr , color = line_col, alpha = 0.2,\
                            label="95% CI")
    axs[2].set_title('')
    axs[2].legend(loc='upper left')

    

    if file_name:
        #sets default dir
        script_dir = os.path.dirname(__file__)
        parent_dir = os.path.abspath(script_dir + "/../")
        directory = parent_dir + '/data/mcmc_chains/img'
        if not os.path.exists(directory):
            os.makedirs(directory)
        #duplicate filename exist append timestamp
        if os.path.isfile(directory+'/'+file_name):
            timestr = time.strftime("%d%m%y_%H%M")
            file_name = file_name.replace('.csv','')
            file_name = file_name + '_' + str(timestr)+ '.csv'

        plt.savefig(directory+'/'+file_name+".png",bbox_inches='tight', dpi=150)
    
    #return forecasted values
    if ret_forecast:
        forecast_nt = []
        pre_val = per_50_arr[0]
        for idx in range(1,len(per_50_arr)):
            cur_val = per_50_arr[idx]
            if pre_val != cur_val:
                forecast_nt.append(x_dis[idx])
            pre_val = per_50_arr[idx]

        return [x_dis,per_32_arr,per_50_arr,per_68_arr,per_95_arr,forecast_nt]

    plt.show()

    return



def uv_exp_plot(max_time,  mu,  alpha,  beta, smp, smp_inj_date = []):
    """plot conditional intesity function for univariate HP. With and without dates. 
    Args:
            max_time (float): widow max value
            mu (float): exogenous variable
            alpha (float): excitation variable
            beta (float): decay variable 
            smp (list of float):  float taus for cond.int
            smp_inj_date, optional (pd.Series: datetime64[ns]): actual dates for x axis. e.g [2018-01-10,2018-02-15...]
    """
    line_col = 'dimgrey' # color= '#1f77b4' - blue
    fig_x = 15
    fig_y = 5
    month_to_days = 30.4167

    # Conditional intensity function
    lda_ar = [mu + np.sum(alpha  * np.exp(-beta * (x - smp[smp < x]))) \
            for x in np.arange(0, max_time, .1)]    

    # plot intensity and counting plots
    fig, axs = plt.subplots(1,1,figsize=(fig_x,fig_y))
    axs.set_ylabel("$\lambda^*(t)$")
    axs.set_xlabel("$t$")

    if len(smp_inj_date) == 0:
        axs.plot(smp, np.ones_like(smp) * mu, 'k.',label="arrivals")
        axs.plot(np.arange(0, max_time, .1), lda_ar, color= line_col ,label="$\lambda^*(t)$")#'b-' ##ff7f0e orange
    else:
        x_rng_tmp = (np.arange(0, max_time, .1) *month_to_days).astype(int)
        x_rng = np.empty(len(x_rng_tmp), dtype='datetime64[D]')

        for i, label in enumerate(x_rng):
            x_rng[i] =  np.datetime64(smp_inj_date.iloc[0]).astype('datetime64[D]') + np.timedelta64(x_rng_tmp[i], 'D')
        
        axs.plot(smp_inj_date[1:], np.ones_like(smp) * mu, 'k.',label="arrivals")
        axs.plot(x_rng, lda_ar, color= line_col ,label="$\lambda^*(t)$")
        axs.set_title('')
        axs.legend(loc='upper left')

    plt.show()
    #plt.savefig(file_path+file_name+"_for.png",bbox_inches='tight', dpi=150)
    return

'''
#test
print('test')
smp = np.array([ 1.97129305,  2.23413212,  2.46411631,  3.15406887,  6.76810612,
        8.37799544,  8.54226986, 12.87911456, 13.40479271, 13.56906713,
       13.93047085, 14.12760016, 14.88326249, 15.34323087, 16.75599088,
       17.41308856, 17.70878252, 19.71293045, 19.94291464, 23.524097  ,
       27.53239286, 27.76237705, 28.18949054, 28.45232962, 30.42362266,
       31.01501058, 31.67210826, 32.09922175])
max_t =34 
#smp = np.array([ 1.97129305,  2.23413212,  2.46411631,  3.15406887])
smp = np.array([0.16427442, 0.55853303, 0.78851722, 0.78861722, 0.91993675,
       1.70845397, 1.77416374, 1.83987351, 2.00414793, 2.00424793,
       2.46411631, 2.46421631, 2.79266515, 2.79276515, 3.05550422,
       3.18692376, 3.41690795, 3.41700795, 3.90973121, 3.94258609])
max_T = 4

'''

#uv_exp_forcast_nt(max_t, 0.1435927990688284,  0.8765741547472792, 0.5875965890054582,\
#                     smp,sim_n = 300,ret_forecast = False, file_name = 'filename')

