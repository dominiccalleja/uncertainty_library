import numpy as np
import pandas as pd
from scipy.linalg import eigh, cholesky
from scipy import stats
from scipy import optimize
from scipy import signal
import time
import pickle
import os
import ray
import sys
__author__ = "Dominic Calleja"
__copyright__ = "Copyright 2020"
__credits__ = "Ander Gray"
__license__ = "MIT"
__version__ = "0.1"
__date__ = '01/4/2020'
__maintainer__ = "Dominic Calleja"
__email__ = "d.calleja@liverpool.ac.uk"
__status__ = "Draft"

print('+===================================+')
print('|               TMCMC               |')
print('|       Bayesian updating tool      |')
print('| Credit: '+__credits__+'        |')
print('|          Version: '+__version__ + 12*' '+' |')
print('|                                   |')
print('| '+__copyright__+' (c) ' + __author__+'|')
print('+===================================+')




class worker_object():
    def __init__(self, prior, prior_sample, likelihood, n_per_layer, n_batches, n_perameters, pickle_path):

        #with open(pickle_path, 'rb') as handle:
        #    GP_data = pickle.load(handle)

        #self.GP = GP_data['GP']
        #self.data_table = data_table
        #self.data = data
        self.likelihood = likelihood
        self.prior = prior
        self.prior_sample = prior_sample
        self.n_perameters = n_perameters

    def _likelihood(self, theta):
        return self.likelihood(theta)

    def _batch_likelihood(self, theta):
        print(theta)
        len_theta = self.n_perameters
        dim = np.asarray(np.shape(theta)) != len_theta
        batch_len = np.shape(theta)[np.where(dim)[0][0]]
        logL = []
        for i in range(batch_len):
            logL.append(self.likelihood(theta[:, i]))
        return logL

    def markov_chain(self, SIGMA_j, THJ, LPJ, a_j1, length, burnin):
        #outputf = open(self.logfile, 'a')
        batch_len = len(LPJ)

        Theta_new = []
        Likelihood_new = []
        accept_proportion = []
        t = time.time()
        for i in range(batch_len):

            Th, L, a = parallel_markov_chain(
                self.prior, self._likelihood, SIGMA_j, THJ[:, i], LPJ[i], a_j1, length[i], burnin=burnin)
            Theta_new.append(Th)
            Likelihood_new.append(L)
            accept_proportion.append(a)
        ap = np.array(accept_proportion)
        return Theta_new, np.asarray(Likelihood_new), accept_proportion


@ray.remote
class tmcmc(worker_object):  # smardda_tmcmc_temperature
    def __init__(self, i, fT, sample_fT, log_fD_T, n_per_layer, Ncores, n_parameters, pickle_path, seq_ind, save_address):
        self.i = i
        super().__init__(fT, sample_fT, log_fD_T,
                         n_per_layer, Ncores, n_parameters, pickle_path)


"""
TMCMC algorithm
"""


def tmcmc_par(log_fD_T, fT, sample_fT, n_per_layer=[], n_batches=[], n_parameters=[], max_n_stages=100, burnin=0, lastburnin=0, beta2=0.01, beta2_b=[], alpha_threshold=[], log_fd_T2=[], process='ray', pickle_path=[], redis_address=[], redis_password=[], num_cpus=[], seq_ind=[], save_address=[], launch_remote_workers=False, logfile='tmcmc_updating_sf_logfile.txt'):
    # Transitional Markov Chain Monte Carlo
    #
    # Usage:
    # [samples_fT_D, fD] = tmcmc(fD_T, fT, sample_from_fT, N)

    # inputs:
    # log_fD_T       = function handle of log(fD_T(t))
    # fT             = function handle of fT(t)
    # sample_from_fT = handle to a function that samples from of fT(t)
    # N              = number of samples of fT_D to generate
    # n_batches      = doubles as n_cores if n_batches>n_cores
    # parallel       = True (default)
    # burnin         = uniform burn in on each iteration for metropolis_hastings
    # lastburnin     = burnin on final iteration to ensure sampling from posterior
    # thining        = intermitten acceptance criterion
    # beta           = is a control parameter that is chosen to balance the
    #                   potential for large MCMC moves while mantaining a
    #                   reasonable rejection rate
    # outputs:
    # samples_fT_D   = samples of fT_D (N x D)
    # log_fD         = log(evidence) = log(normalization constant)
    #
    # This program implements a method described in:
    # Ching, J. and Chen, Y. (2007). "Transitional Markov Chain Monte Carlo
    # Method for Bayesian Model Updating, Model Class Selection, and Model
    # Averaging." J. Eng. Mech., 133(7), 816-832.

  # Square of scaling parameter(MH algorithm)
    #Preallocation of number of stages

    # PRE_ALLOCATION
    Th_j = []  # cell(max_n_stages, 1)
    alpha_j = np.zeros(max_n_stages)
    Lp_j = []
    Log_S_j = np.zeros(max_n_stages)
    wn_j = []
    n_per_batch = n_per_layer/n_batches
    acceptance = []
    t = time.time()
    #logfile = 'tmcmc_updating_sf_logfile.txt'
    if seq_ind == 0 or launch_remote_workers:
        copyR(logfile)
    outputf = open(logfile, 'a')

    # import libraries
    # multiprocessing setup
    #if parallel:
    import pathos.multiprocessing as mp

    Ncores = min(mp.cpu_count(), n_batches)
    print('TMCMC is running on {} cores'.format(Ncores))
    outputf.write('TMCMC is running on {} cores\n'.format(Ncores))
    outputf.write('seq ind : {}\n'.format(seq_ind))

    j = 0
    alpha_j[0] = 0
    # sample from prior
    Th_j.append(sample_fT(n_per_layer))
    Th0 = Th_j[0]

    if process == 'pathos':
        #par likelihood
        p = mp.Pool(Ncores)
        print('Executing processing with pathon. WARNING: if model is heavy the initialisation time may be very long!')
        def par_like(theta): return log_fD_T(Th0[:, theta])
        Lp0 = p.map(par_like, range(n_per_layer))
        Lp0 = np.array(Lp0)
        Lp0 = np.reshape(Lp0, [n_per_layer])
    if process == 'ray':
        if seq_ind == 0 or launch_remote_workers:
            if redis_password:
                # , _redis_password=redis_password)#os.environ["ip_head"]
                ray.init(address="auto")
                #ray.init(address='auto', _redis_password=redis_password)
                Ncores = num_cpus
            else:
                ray.init(num_cpus=Ncores, lru_evict=True)
        print('Executing processing with Ray. WARNING: if model is heavy there may be memory issues!')
        t_like = time.time()
        print("RAY NODES: {}".format(ray.nodes))
        actors = [tmcmc.remote(
            i, fT, sample_fT, log_fD_T, n_per_layer, Ncores, n_parameters, pickle_path, seq_ind, save_address) for i in range(Ncores)]
        theta_b = np.split(Th0, Ncores, axis=1)
        Lp0 = ray.get([actor._batch_likelihood.remote(
            theta_b[a][:, range(np.shape(theta_b[0])[1])]) for a, actor in enumerate(actors)])

        try:
            Lp0 = np.reshape(Lp0, [n_per_layer])
        except:
            Lp0 = np.concatenate(Lp0)
        print('Completed likelihood evaluation. Time Elapsed : {}'.format(
            timestamp_format(t_like, time.time())))
        outputf.write('Completed likelihood evaluation. Time Elapsed : {}\n'.format(
            timestamp_format(t_like, time.time())))

    Lp0[Lp0 == -np.inf] = -1E5
    Lp_j.append(Lp0)
    outputf.close()
    while alpha_j[j] < 1:
        outputf = open(logfile, 'a')
        if j+1 == max_n_stages:
            print('Reached limit of stages {}. Terminating run without convergence'.format(
                max_n_stages))
            outputf.write(
                'Reached limit of stages {}. Terminating run without convergence'.format(max_n_stages))
            break

        t1 = time.time()
        print('TMCMC: Iteration j = {}'.format(j))
        outputf.write('TMCMC: Iteration j = {}\n'.format(j))
        # Find tempering parameter
        print('Computing the tempering ...')
        outputf.write('Computing the tempering ...\n')

        alpha_j[j+1], Lp_adjust = calculate_pj_alpha(Lp_j[j], alpha_j[j])
        print('TMCMC: Iteration j = {}, pj1 = {}'.format(j, alpha_j[j+1]))
        print('Computing the weights ...')
        outputf.write(
            'TMCMC: Iteration j = {}, pj1 = {}\n'.format(j, alpha_j[j+1]))
        outputf.write('Computing the weights ...\n')
        #Adjusted weights
        w_j = np.exp((alpha_j[j+1]-alpha_j[j])*(Lp_j[j]-Lp_adjust))

        print('Computing the evidence ...')
        outputf.write('Computing the evidence ...\n')
        #Log-evidence of j-th intermediate distribution
        Log_S_j[j] = np.log(np.mean(np.exp(
            (Lp_j[j]-Lp_adjust)*(alpha_j[j+1]-alpha_j[j]))))+(alpha_j[j+1]-alpha_j[j])*Lp_adjust

        #Normalized weights
        wn_j.append(w_j/(np.sum(w_j)))
        print('Computing the covariance ...')
        outputf.write('Computing the covariance ...\n')

        # Weighted mean and coviariance
        if alpha_j[j+1] > alpha_threshold:
            beta2 = beta2_b
            #log_fD_T = log_fd_T2
            print('Adaptive likelihood: Switching Beta')

        Th_wm = np.matrix(Th_j[j]) * np.matrix(wn_j[j]).T
        SIGMA_j = np.zeros([n_parameters, n_parameters])
        for l in range(n_per_layer):
            SIGMA_j = SIGMA_j + beta2 * \
                wn_j[j][l] * (Th_j[j][:, l] - Th_wm.T).T * \
                (Th_j[j][:, l] - Th_wm.T)
        SIGMA_j = (SIGMA_j.T + SIGMA_j)/2
        # Metropolis Hastings
        print('Inititialising Metropolis Hastings ...')
        outputf.write('Inititialising Metropolis Hastings ...\n')
        wn_j_csum = np.cumsum(wn_j[j])
        n_mc, seed_index, mkchain_ind = markov_chain_seed(
            wn_j_csum, n_per_layer)
        #print(seed_index)
        lengths = np.zeros(np.shape(seed_index))

        for i_mc in range(lengths.size):
            lengths[i_mc] = np.sum(seed_index[i_mc] == mkchain_ind)

        # Improve posterior sampling:
        if alpha_j[j+1] == 1:
            burnin = lastburnin

        # Preallocation
        a_j1 = alpha_j[j+1]
        THJ = Th_j[j][:, seed_index]
        LPJ = np.array(Lp_j[j])[seed_index]
        results = []

        print('Markov chains ...')
        outputf.write('Markov chains ...\n')
        outputf.close()
        outputf = open(logfile, 'a')
        if process == 'pathos':
            print('Executing mc with pathos')

            def func(t): return parallel_markov_chain(fT, log_fD_T, SIGMA_j,
                                                      THJ[:, t], LPJ[t], a_j1, lengths[t], burnin=burnin)
            results = p.map(func, range(n_mc))
            print('Formatting outputs ...')
            outputf.write('Formatting outputs ...\n')

            Th_j_tmp = results[0][0]
            Lp_j_tmp = results[0][1]
            acc_rate = []
            acc_rate.append(results[0][2])
            for i in range(1, n_mc):
                Th_j_tmp = np.concatenate((Th_j_tmp, results[i][0]), axis=0)
                Lp_j_tmp = np.concatenate((np.reshape(Lp_j_tmp, [len(Lp_j_tmp), 1]), np.reshape(
                    results[i][1], [len(results[i][1]), 1])), axis=0)
                acc_rate.append(results[i][2])
            Th_j.append(Th_j_tmp.T)
            Lp_j.append(np.squeeze(np.array(Lp_j_tmp, dtype=float)))
            acceptance.append(acc_rate)
        if process == 'ray':
            print('Executing mc with ray')
            #SIGMA_j, THJ, LPJ, a_j1, length, burnin
            ind_mc = np.asarray(range(n_mc))
            batch_ind = np.array_split(ind_mc, Ncores)
            #return SIGMA_j, THJ, LPJ, batch_ind, actors
            results = ray.get([actor.markov_chain.remote(SIGMA_j, THJ[:, batch_ind[a]],
                                                         LPJ[batch_ind[a]], a_j1, lengths[batch_ind[a]], burnin) for a, actor in enumerate(actors)])

            print('Formatting outputs ...')
            outputf.write('Formatting outputs ...\n')
            r = []
            [r.append(np.concatenate(results[i][0], axis=0))
             for i in range(len(results))]
            Th_j_tmp = np.concatenate(r, axis=0)
            l = []
            [l.append(np.hstack(results[i][1])) for i in range(len(results))]
            Lp_j_tmp = np.concatenate(l)
            Lp_j_tmp = np.array(Lp_j_tmp, dtype=float)
            a = []
            [a.append(results[i][2]) for i in range(len(results))]
            acc_rate = np.concatenate(a, axis=0)

            Th_j.append(Th_j_tmp.T)
            Lp_j.append(Lp_j_tmp)
            acceptance.append(acc_rate)

        print('TMCMC: Iteration j = {} complete. Time Elapsed : {} \n\n'.format(
            j, timestamp_format(t1, time.time())))
        outputf.write('TMCMC: Iteration j = {} complete. Time Elapsed : {} \n\n'.format(
            j, timestamp_format(t1, time.time())))
        outputf.write('+'+'='*77+'+ \n')
        j = j+1
        outputf.close()
    outputf = open(logfile, 'a')
    m = j
    print('TMCMC Complete: Evaluated posterior in {} iterations. Time Elapsed {} '.format(
        m, timestamp_format(t, time.time())))
    outputf.write('TMCMC Complete: Evaluated posterior in {} iterations. Time Elapsed {} '.format(
        m, timestamp_format(t, time.time())))
    outputf.write('+'+'='*77+'+ \n')

    Th_posterior = Th_j[m]
    Lp_posterior = Lp_j[m]
    #ray.shutdown()
    outputf.close()
    return Th_posterior,  Lp_posterior, acceptance, Th_j, Lp_j


"""
markov for par_tmcmc
"""


def parallel_markov_chain(fT, log_fD_T, SIGMA_j, THJ, LPJ, a_j1, length, burnin=None):
    Th_lead = THJ
    Lp_lead = LPJ
    Th_new = []
    Lp_new = []
    a = 0
    time_stopper = time.time()
    for l in range(0-burnin, length.astype(int)):
        #------------------------------------------------------------------
        # Candidate sample generation (normal over feasible space)
        #------------------------------------------------------------------
        while True:
            Th_cand = stats.multivariate_normal.rvs(Th_lead, SIGMA_j)

            t_delta = time.time()-time_stopper

            if t_delta/60 > 45:
                print('\n\n\n WARNING: GOT VERY STUCK!!!!!!!! \n\n\n')

                Th_lead = np.ones(np.shape(Th_lead))*np.random.uniform(2, 8)
                print('moving sample to {}'.format(Th_lead))

            if not fT(Th_cand) == 0:
                break
        #------------------------------------------------------------------
        # Log-likelihood of candidate sample
        #------------------------------------------------------------------
        if fT(Th_cand) == 0:
            GAMMA = 0
            Lp_cand = Lp_lead
        else:
            Lp_cand = log_fD_T(Th_cand)
            GAMMA = np.exp(a_j1*(Lp_cand - Lp_lead))*fT(Th_cand)/fT(Th_lead)
        #------------------------------------------------------------------
        # Rejection step
        #------------------------------------------------------------------
        thresh = np.random.rand()
        if thresh <= min(1, GAMMA):
            Th_lead = Th_cand
            Lp_lead = Lp_cand
            a = a+1
        if l >= 0:
            if thresh <= min(1, GAMMA):
                Th_new.append(Th_cand)
                Lp_new.append(Lp_cand)
            else:
                Th_new.append(Th_lead)
                Lp_new.append(Lp_lead)
    return np.array(Th_new), np.reshape(np.asarray(Lp_new), [length.astype(int)]), np.array(a / (burnin+length))


def markov_chain_seed(wn_j_csum, Nm):
    # Definition of Markov chains: seed sample
    mkchain_ind = np.zeros(Nm)
    for i_mc in range(Nm):
        #while True:
        mkchain_ind[i_mc] = np.argwhere(np.random.rand() < wn_j_csum)[0]

    seed_index = np.unique(mkchain_ind).astype(int)
    N_Mc = np.size(seed_index)
    return N_Mc, seed_index, mkchain_ind


"""
Tempering parameter
"""


def calculate_pj_alpha(log_fD_T_thetaj, alpha_j):
    #----------------------------------------------------------------------
    # choose pj    (Bisection method)
    #----------------------------------------------------------------------
    low_alpha = alpha_j
    up_alpha = 2
    Lp_adjust = np.max(log_fD_T_thetaj)

    while ((up_alpha - low_alpha)/((up_alpha + low_alpha)/2)) > 1e-6:
        x1 = (up_alpha + low_alpha)/2
        wj_test = np.exp((x1-alpha_j)*(log_fD_T_thetaj-Lp_adjust))
        cov_w = np.std(wj_test)/np.mean(wj_test)
        if cov_w > 1:
            up_alpha = x1
        else:
            low_alpha = x1
    alpha_j = min(1, x1)
    return alpha_j, Lp_adjust


"""
log file
"""


def timestamp_format(t0, t1):
    t_delta = t1-t0

    if t_delta/60 < 1:
        time = '{:.3f} (sec)'.format(t_delta)
    elif t_delta/60 > 1 and t_delta/60/60 < 1:
        time = '{:.3f} (min)'.format(t_delta/60)
    else:
        time = '{:.3f} (hrs)'.format(t_delta/60/60)

    return time


def copyR(logfile):
    """Print copyright information to file."""
    outputf = open(logfile, 'a')
    outputf.write('+'+'='*77+'+ \n')
    tl = 'Updating4Smardda'
    sp1 = (77-len(tl))//2
    sp2 = 77-sp1-len(tl)
    outputf.write('|'+' '*sp1+tl+' '*sp2+'|' + '\n')
    tl = 'Bayesian updating tools for Smardda'
    sp1 = (77-len(tl))//2
    sp2 = 77-sp1-len(tl)
    outputf.write('|'+' '*sp1+tl+' '*sp2+'|' + '\n')
    tl = ' '
    sp1 = (77-len(tl))//2
    sp2 = 77-sp1-len(tl)
    outputf.write('|'+' '*sp1+tl+' '*sp2+'|' + '\n')
    tl = ' Version: '+__version__
    sp1 = (77-len(tl))//2
    sp2 = 77-sp1-len(tl)
    outputf.write('|'+' '*sp1+tl+' '*sp2+'|' + '\n')
    outputf.write('|'+' '*77+'| \n')
    tl = __copyright__+' (c) ' + __author__
    sp1 = (77-len(tl))//2
    sp2 = 77-sp1-len(tl)
    outputf.write('|'+' '*sp1+tl+' '*sp2+'|' + '\n')
    outputf.write('+'+'='*77+'+' + '\n')
    outputf.write('\n')
    outputf.close()
    return

"""
Area metric
"""


def area_metric_robust(D1, D2):
    """
    #   Returns the stochastic distance between two data
    #   sets, using the area metric (horizontal integral between their ecdfs)
    #
    #   As described in: "Validation of imprecise probability models" by S.
    #   Ferson et al. Computes the area between two ECDFs
    #
    #                  By Marco De Angelis, (adapted for python Dominic Calleja)
    #                     University of Liverpool by The Liverpool Git Pushers
    """

    if np.size(D1) > np.size(D2):
        d1 = D2
        d2 = D1
    else:
        d1 = D1
        d2 = D2      # D1 will always be the larger data set

    Pxs, xs = ecdf(d1)            # Compute the ecdf of the data sets
    Pys, ys = ecdf(d2)

    Pys_eqx = Pxs
    Pys_pure = Pys[0:-1]  # this does not work with a single datum
    Pall = np.sort(np.append(Pys_eqx, Pys_pure))

    ys_eq_all = np.zeros(len(Pall))
    ys_eq_all[0] = ys[0]
    ys_eq_all[-1] = ys[-1]
    for k in range(1, len(Pall)-1):
        ys_eq_all[k] = interpCDF_2(ys, Pys, Pall[k])

    xs_eq_all = np.zeros(len(Pall))
    xs_eq_all[0] = xs[0]
    xs_eq_all[-1] = xs[-1]
    for k in range(1, len(Pall)-1):
        xs_eq_all[k] = interpCDF_2(xs, Pxs, Pall[k])

    diff_all_s = abs(ys_eq_all-xs_eq_all)
    diff_all_s = diff_all_s[range(1, len(diff_all_s))]
    diff_all_p = np.diff(Pall)
    area = np.matrix(diff_all_p) * np.matrix(diff_all_s).T

    return np.array(area)[0]



def interpCDF_2(xd,yd,pvalue):
    """
    %INTERPCDF Summary of this function goes here
    %   Detailed explanation goes here
    %
    % .
    % . by The Liverpool Git Pushers
    """
    # [yd,xd]=ecdf(data)
    beforr = np.zeros(len(yd))
    beforr = np.diff(pvalue <= yd) ==1
    beforrr = np.append(0,beforr[:])
    if pvalue==0:
        xvalue = xd[1]
    else:
        xvalue = xd[beforrr==1]

    outputArg1 = xvalue

    return outputArg1


def ecdf(x):
    xs = np.sort(x)
    #xs = np.append(xs,xs[-1])
    n = xs.size
    y = np.linspace(0,1,n)
    #np.arange(1, n+1) / n
    #xs = np.append(xs[0],xs)
    #ps =
    return [y,xs]


if __name__ == '__main__':

    D = 5
    covmatReal = np.identity(2)
    muReal1 = [D,D,D,D,D,D,D,D]               #   Top octant
    muReal2 = [-D,-D,-D,-D,-D,-D,-D,-D]        #   Bottom octant


    def likelihood(theta,Data,D,model):
        y = model(theta)
        area = []
        p = []
        for i in range(D):
            area.append(area_metric_robust(y[i],Data[i]))
            p.append(-(1/0.1)**2 * (area[i]**2))

        if np.isinf(p).any():
            p = -1e10
        return np.sum(p)

    def model_simple(inp,n_model):
        mu1 = inp[0]
        mu2 = inp[1]
        mu3 = inp[2]
        mu4 = inp[3]
        sig1 = inp[4]
        sig2 = inp[5]

        x1 = (np.random.normal(mu1, sig1, size=int(n_model/2)))
        x2 = (np.random.normal(mu2, sig1, size=int(n_model/2)))
        x = np.concatenate([x1,x2],axis=0)

        y1 = (np.random.normal(mu3, sig2, size=int(n_model/2)))
        y2 = (np.random.normal(mu4, sig2, size=int(n_model/2)))
        y = np.concatenate([y1,y2],axis=0)

        return [x,y]

    def prior_rnd_simple(N, limit):
        means = np.random.uniform(limit[0][0],limit[0][1] ,size=[4,N])
        std_devs  = np.random.uniform(limit[1][0],limit[1][1] ,size=[2,N])
        return np.concatenate([means,std_devs],axis=0) #scales

    def prior_pdf_simple(x, limit):
        P = (stats.uniform.pdf(x[0], loc = limit[0][0],scale = (limit[0][1]-limit[0][0])) * # mean
            stats.uniform.pdf(x[1], loc = limit[0][0],scale = (limit[0][1]-limit[0][0])) * # mean
            stats.uniform.pdf(x[2], loc = limit[0][0],scale = (limit[0][1]-limit[0][0])) * # mean
            stats.uniform.pdf(x[3], loc = limit[0][0],scale = (limit[0][1]-limit[0][0])) * # mean
            stats.uniform.pdf(x[4], loc = limit[1][0],scale = (limit[1][1]-limit[1][0])) * # std
            stats.uniform.pdf(x[5], loc = limit[1][0],scale = (limit[1][1]-limit[1][0]))) # std
        return P
    limit = [[-20, 15],[0.2, 9]]

    DATA = model_simple([-6, 10, -6, 7, 0.4, 3], 100)

    def priorR(N): return prior_rnd_simple(N,limit)

    def prior(x): return prior_pdf_simple(x, limit)

    def model(theta): return model_simple(theta,1000)


    def log_fD_T(theta): return likelihood(theta, DATA, 2, model)


    Th, Lp, acceptance, Th_j1, Lp_j1 = tmcmc_par(log_fD_T, prior, priorR, n_per_layer=400, n_batches=10, n_parameters=6, max_n_stages=40,
                                                    burnin=0, lastburnin=0, beta2=0.3, beta2_b=[], process='ray', pickle_path=[], seq_ind=0, save_address=[])
