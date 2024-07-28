#Script used with report showing model behaviours
#given scale of alpha and relationship to beta.
import numpy as np
import abc 
from matplotlib import pyplot as plt

#ogata requirments
import random 
import math

#for the max log likelihood
from scipy.optimize import minimize

#for big 0
import timeit
#from datetime import datetime


def inhomogenous_poisson_process(T=12,plot=True):
    """
    Implements inhomogeneous Poisson process with intensity function,
    λ(t) = cos(t) + 1. Requires to edit functionally manually.

    :param T: (optionally) the maximum time
    :param plot: (boolean) plot 

    :return: 1-d numpy ndarray accepted samples
    :return: 1-d numpy ndarray rejected samples
    """

    t = 0
    pt_inhom,pt_homo,reject = [],[],[]
    delta = 0.1

    m = max(np.asarray([np.cos(x) +1  \
        for x in np.arange(t, T, delta)]))
    
    while t < T:
        r1 = random.uniform(0,1)
        U = -math.log(r1)/m  #exponential simulated

        t = t + U 
        pt_homo.append([t,0]) #homogenous points 

        r2 = random.uniform(0,m)
        func = np.cos(t) + 1

        if r2 <= func:
            pt_inhom.append([t,r2])
        else:
            reject.append([t,r2])

    if plot==True:
        #fig, axs = plt.subplots(1)
        fig, axs = plt.subplots(1,figsize=(15,2) )
        x_min, x_max = 0, T
        w = np.linspace(0,T,150)
        func = np.cos(w) +1
        axs.set_ylabel("$\lambda^*(t)$")
        axs.set_xlabel("$t$")
        axs.plot(w, func, '#1f77b4',label='$\lambda(t)$')
        axs.hlines(m,0,T,color='#1f77b4',linestyles ='dashed',label='M')
        axs.scatter([row[0] for row in pt_inhom],[row[1] for row in pt_inhom],facecolors='none', edgecolors='black',s=60,label='inhomogeneous')
        axs.vlines([row[0] for row in pt_inhom],0,[row[1] for row in pt_inhom],color='black', linestyles ='dotted',alpha=0.5)
        axs.scatter([row[0] for row in reject],[row[1] for row in reject],color='red',marker='x',s=60,label='rejected')
        axs.vlines([row[0] for row in reject],0,[row[1] for row in reject],color='red',linestyles ='dotted',alpha=0.5)
        axs.scatter([row[0] for row in pt_homo],[row[1] for row in pt_homo],color='black',s=20,label='homogeneous')
        #axs.set_title('Inhomogeneous Poisson process with bounded intensity function') 
        axs.set_ylim(ymin=-0.05)
        axs.set(xlim=(x_min, x_max))
        axs.legend(loc="upper right")
        plt.show()

    return pt_inhom, reject



def uu() :
    return random.uniform(0, 1)


def _get_offspring( t,  alpha,  beta,  T):

    N = np.random.poisson(alpha)
    os = np.empty(shape=(N,), dtype=np.float)  # offsprings

    for j in range(N):
        tt = -math.log(uu()) / beta + t
        os[j] = tt

    return os[os < T]


def uv_exp_sample_branching( T,  mu,  alpha,  beta):
    """
    Implement a branching sampler for a univariate exponential HP, taking advantage of the
    cluster process representation. As pointed out by Moller and Rasmussen (2005), this is an approximate sampler
    and suffers from edge effects.
    """

    birth = np.array([])
    immag = np.array([])

    imm_count = np.random.poisson(mu * T)
    curr_gen = np.random.rand(imm_count) * T
    P = np.array([])

    while len(curr_gen) > 0:
        P = np.concatenate([P, curr_gen])
        offsprings = []
        for k in curr_gen:
            v = _get_offspring(k, alpha, beta, T)
            offsprings.append(v)

        curr_gen = np.concatenate(offsprings)

    P.sort(kind="mergesort")

    return P


def uv_exp_sample_ogata( T,  mu,  alpha,  beta,  phi=0):
    """`
    Implements Ogata's modified thinning algorithm for sampling from a univariate Hawkes process
    with exponential decay.

    :param T: the maximum time
    :param mu: the exogenous intensity
    :param alpha: the infectivity factor alpha
    :param beta: intensity parameter of the delay density
    :param phi: (optionally) the starting phi, the running sum of exponential differences specifying the history until
    a certain time, thus making it possible to take conditional samples

    :return: 1-d numpy ndarray of samples
    """

    t = 0.
    ed = 0.
    d = 0.
    lda = 0.
    td = []
    rj = []

    while t < T:
        M = mu + alpha * beta * (1 + phi) #set upper bound

        r1 = random.uniform(0, 1) #uniform 1
        r2 = random.uniform(0, 1)

        E = -math.log(r1) / M
        t = t + E #time step

        ed = math.exp(-beta * (E + d))
        lda = mu + alpha * beta * ed * (1 + phi)

        if t < T and r2 * M <= lda:
            td.append(t)   # accepted cadidate
            phi = ed * (1 + phi)
            d = 0
        else:
            if t < T:
                rj.append(t)  #rejected cadidate
            d = d + E

    res = np.empty(len(td),np.float)
    for j in range(len(res)):
        res[j] = td[j]

    #keeping rejected candidates
    rej = np.empty(len(rj),np.float)
    for k in range(len(rej)):
        rej[k] = rj[k]   

    return res #,rej


def uv_exp_fit_em_base(t,  T, maxiter=500, reltol=1e-5):
    """
    Fit a univariate Hawkes process with exponential decay function using the Expectation-Maximization
    algorithm. The algorithm exploits the memoryless property of the delay density to compute the E-step
    in linear time. Due to the Poisson cluster property of HP, the M-step is in constant time.

    :param t: Bounded finite sample of the process up to time T. 1-d ndarray of timestamps
    :param T: the maximum time
    :param maxiter: int, maximum number of EM iterations
    :param reltol: double, the relative improvement tolerance to stop the algorithm
    :return: tuple, (final log likelihood, (mu, alpha, theta))
    """

    t0 = t[0]
    odll = 0.
    odll_p = 0.
    i, j = 0, 0
    N = len(t)

    mu = N * 0.8 * (1 + (uu() - .5) / 10.) / T
    alpha = 0.2 + (uu() - .5) / 10.
    theta = mu * (1 + (uu() - .5) / 10.)

    odll_p = uv_exp_ll(t, mu, alpha, theta, T)

    for j in range(maxiter):

        # E-step

        # initialize accumulators
        phi, ga = 0, 0

        # initialize ESS
        E1 = 1. / mu
        E2, E3 = 0, 0
        C1, C2 = 0, 0


        C1 += 1 - np.exp(-theta * (T - t0))
        C2 += (T - t0) * np.exp(- theta * (T - t0))

        for i in range(1, N):
            ti = t[i]
            d = ti - t[i-1] + 1e-15
            r = T - ti + 1e-15

            ed = np.exp(-theta * d)  # log of the exp difference exp(-theta * d)
            er = np.exp(-theta * r)  # log of the exp difference of time remaining (for the compensator)

            ga = ed * (d * (1 + phi) + ga)
            phi = ed * (1 + phi)

            Z = mu + alpha * theta * phi  # Missing values(latent) - jw
            atz = alpha * theta / Z

            # collect ESS - expected sufficient statistics
            E1 += 1. / Z
            E2 += atz * phi
            E3 += atz * ga

            C1 += 1 - er
            C2 += r * er

        # M-step

        mu = mu * E1 / T
        theta = E2 / (alpha * C2 + E3)
        alpha = E2 / C1

        # calculate observed data log likelihood

        odll = uv_exp_ll(t, mu, alpha, theta, T)
        relimp = (odll - odll_p) / abs(odll_p)  # relative improvement
        if relimp < -1e-5:
            raise Exception("Convergence problem, the log likelihood did not increase")
        elif relimp < reltol:
            break
        odll_p = odll

    #return (mu, alpha, theta)
    return odll, (mu, alpha, theta), j


def uv_exp_ll_grad(t,  mu,  alpha,  beta,  T):
    """
    Calculate the gradient of the likelihood function w.r.t. parameters mu, alpha, beta

    :param t: Bounded finite sample of the process up to time T. 1-d ndarray of timestamps
    :param mu: the exogenous intensity
    :param alpha: the infectivity factor alpha
    :param beta: intensity parameter of the delay density
    :param T: the maximum time
    :return: the gradient as a numpy.array of shape (3,). Gradients w.r.t. mu, alpha, beta respectively
    """

    phi, nphi = 0, 0
    Calpha,Cbeta = 0,0
    nmu , nalpha , nbeta = 0.,0.,0.
    N = len(t)
    j = 0
    d,r = 0., 0.

    nmu = 1/ mu
    Calpha = 1 - math.exp(-beta * (T - t[0]))
    Cbeta = alpha * (T - t[0]) * math.exp(-beta * (T - t[0]))

    for j in range(N-1):
        d = t[j+1] - t[j]
        r = T - t[j+1]

        ed = math.exp(-beta * d)
        #F = 1 - math.exp(-beta * r)
        try:
            F = 1 - math.exp(-beta * r)
        except: # number is too large
            F = 1 - math.exp(50)

        nphi = ed * (d * (1 + phi) + nphi)
        phi = ed * (1 + phi)
        lda = mu + alpha * beta * phi

        nmu = nmu + 1. / lda
        nalpha = nalpha + beta * phi / lda
        nbeta = nbeta + alpha * (phi - beta * nphi) / lda

        Calpha = Calpha + F
        Cbeta = Cbeta + alpha * r * (1 - F)

    return np.array([nmu - T, nalpha - Calpha, nbeta - Cbeta])


def uv_exp_ll(t, mu, alpha, beta, T):
    """
    Likelihood of a univariate Hawkes process with exponential decay.

    :param t: Bounded finite sample of the process up to time T. 1-d ndarray of timestamps
    :param mu: the exogenous intensity
    :param alpha: the infectivity factor alpha
    :param beta: intensity parameter of the delay density
    :param T: the maximum time
    :return: the log likelihood
    """

    phi = 0.
    lComp = -mu * T
    lJ = 0
    N = t.shape[0]
    j = 0

    lComp -= alpha * (1 - math.exp(-beta * (T - t[0])))
    lJ = math.log(mu)

    for j in range(N-1):
        d = t[j+1] - t[j]
        r = T - t[j+1]

        ed = math.exp(-beta * d)  # exp_diff
        try:
            F = 1 - math.exp(-beta * r)
        except: # number is too large
            F = 1 - math.exp(50)

        phi = ed * (1 + phi) # the running sum of exponential differences specifying the history a certain time, thus making it possible to take conditional samples.
        lda = mu + alpha * beta * phi

        lJ = lJ + math.log(lda)
        lComp -= alpha * F

    return lJ + lComp


def _fit_grad_desc(t, T=None):
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

        minres = minimize(lambda x: -uv_exp_ll(t, x[0], x[1], x[2], T),
                        x0=np.array([mu0, a0, th0]),
                        jac=lambda x: -uv_exp_ll_grad(t, x[0], x[1], x[2], T),
                        bounds=[(1e-5, None), (1e-5, 1), (1e-5, None)],
                        #method="BFGS", options={"disp": False,"gtol": 1e-8})
                        method="L-BFGS-B", options={"disp": False, "ftol": 1e-10, "gtol": 1e-8})

        ress.append(minres)
        mu, a, _ = minres.x

        # take the unconditional mean and see if it makes sense
        Napprox = mu * T / (1 - a)   
        if abs(Napprox - N)/N < .01:  # if the approximation error is in range, break
            break

    return mu, a, _ # remove this
