from scipy.special import erfinv
from scipy.linalg import eigh, cholesky
#from sklearn.neighbors import KernelDensity
import statsmodels.api as sm
from statsmodels.distributions.mixture_rvs import mixture_rvs
from scipy.stats import gaussian_kde
import sys 


import time
import numpy as np
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

import copy
import seaborn as sns
import matplotlib.gridspec as gridspec
mpl.rcParams.update({'font.size': 15})

import scipy.signal as sig
import scipy.stats as stats

try: 
    from smardda_stochastic_model_special import *
except:
    print('No smardda specials imported! \n\t Dont worry these are only for a specific application')

"""
Basic library USAGE

1 - first generate or input some bivariate data. Here A and B

    A = np.random.uniform(0, 100, 1000)
    B = np.random.uniform(0, 100, 1000)
    plt.scatter(A, B)
    plt.show()


2 - construct you joint probability model
first make a stochastic model object telling it the input names
    SM = Stochastic_model(['A', 'B'])

then pass in key value pairs with the object data
    SM.generate_marginal(A=A, B=B)

3 - specify the copula
choose a copula parameter
    theta = 5
pass the copula parameter and choose from the list of availiable copulas
    SM.choose_copula(theta, copula_choice='Clayton')

4 - sample the copula
    X = SM.sample(1000)

5 - plot the independent vs correlated copula samples in real space
    plt.scatter(A, B)
    plt.scatter(X[:, 0], X[:, 1])
    plt.show()

6 - visulaise copula CDF

    SM.plot_copula_cdf()
"""

"""
Class for univariate probability methods 
"""


class PDF():
    def __init__(self, Data):
        self.data = Data
        self.P, self.X = ecdf(Data)

        self.label = ''
        self.Description = ''
        self.units = ''
    """
    def __new__(self, *args, **kwargs):
        return self.data.super().__new__(cls, *args, **kwargs)
    """

    def describe_object(self):
        self._axis_label = '{} ({})'.format(self.label, self.units)

    def inverse_transform(self, P):
        if not hasattr(self, '_kde'):
            self.fit_kde()
        # np.interp(X, self.P, self.X)
        return np.interp(P, self._P_kde,  self._X_kde)

    def ppf(self, X):
        if not hasattr(self, '_kde'):
            self.fit_kde()
        # np.interp(X, self.X, self.P)
        return np.interp(X, self._X_kde, self._P_kde)

    def inverse_transform_ecdf(self, P):
        return np.interp(P, self.P, self.X)

    def ppf_ecdf(self, X):
        return np.interp(X, self.X, self.P)

    def sample(self, N):
        U = np.random.uniform(0, 1, N)
        #self.inverse_transform(U)
        return self.inverse_transform(U)

    def arg_sample(self, N, return_value=True):
        U = np.random.uniform(0, 1, N)
        xi = np.zeros(N)
        y = np.zeros(N)
        for i in range(N):
            x = self.inverse_transform(U[i])
            xi[i] = np.argmin(abs(self.data-x))
            y[i] = self.data[int(xi[i])]
        if return_value:
            return xi, y
        return xi

    def fit_kde(self):
        kde = sm.nonparametric.KDEUnivariate(self.data)
        kde.fit(fft=True)
        self._kde = kde
        self._P_kde = kde.cdf
        self._X_kde = kde.support

        """
        self._kde = KernelDensity(
            kernel='gaussian', bandwidth=0.75).fit(self.data.reshape(-1,1))
        """
        return kde

    def confidence_distribution(self, alpha=0.95, n_interp=1000, xlim=[-1, 1]):
        L, R, X = confidence_limits_distribution(
            self.data, alpha, n_interp=n_interp, x_lim=xlim)
        self._confidence_limits = np.concatenate([[X], [L], [R]])
        return X, L, R

    def confidence_mean(self, alpha=0.95):
        self._confidence_mean = confidence_interval_mean(
            self.data, confidence_interval=alpha)
        return self._confidence_mean

    def interp_condfidence_distribution(self, U, alpha=0.95):
        N = np.size(self.data)
        if len(self.data) > 1E3/2:
            print(
                'WARNING: Confidence limit interpolation too small. Increasing interpolant')
            self.confidence_distribution(
                alpha=alpha, n_interp=2*len(self.data))

        b_l = np.interp(
            U, self._confidence_limits[1, :], self._confidence_limits[0, :])
        b_r = np.interp(
            U, self._confidence_limits[2, :], self._confidence_limits[0, :])
        if b_l <= self._confidence_limits[0, :].min():
            b_l = np.inf
        if b_r >= self._confidence_limits[0, :].max():
            b_r = np.inf
        return b_l, b_r

    def plot_kde(self, n_bins=30, save_address=[]):
        fig = plt.figure(figsize=(12, 9))
        sns.distplot(self.data, n_bins)
        if save_address:
            fig.savefig(save_address)
        plt.show()

    def plot_ecdf(self, save_address=[]):
        self.describe_object()

        fig = plt.figure(figsize=(9, 9))
        plt.step(self.X, self.P)
        plt.xlabel(self._axis_label)
        plt.ylabel('P(x)')
        if save_address:
            fig.savefig(save_address)
        plt.show()

    def plot_confidence_distribution(self, alpha=0.95, n_interp=200, xlim=[-1, 1]):
        self.describe_object()

        L, R, X = confidence_limits_distribution(
            self.data, alpha, n_interp=n_interp, plot=True, label=self._axis_label, x_lim=xlim)
        plt.show()

    def plot_multivariate_data(self, Y0, Y1, label_0='', label_1='', save_address=[]):
        self.describe_object()

        fig = plt.figure(figsize=(12, 9))
        G = gridspec.GridSpec(4, 4)
        ax = plt.subplot(G[:, :2])
        ax = sns.distplot(self.data, 30)
        ax.set_xlabel(self._axis_label)

        ax.set_xlim([0, 125])
        ax2 = plt.subplot(G[2:, 2:])
        ax2.scatter(self.data, Y0, marker='+', s=50)
        ax2.set_ylabel(label_0)
        ax2.set_xlabel(self._axis_label)
        ax2.set_xlim([0, 125])

        ax1 = plt.subplot(G[:2, 2:], sharex=ax2)
        ax1.scatter(self.data, Y1, marker='+', s=50)
        ax1.set_ylabel(label_1)
        ax1.set_xlabel(self._axis_label)
        ax1.set_xlim([0, 125])
        plt.tight_layout()
        if save_address:
            fig.savefig(save_address)
        plt.show()

"""
Class for bivariate stochastic model defined by the marginals and an archemedian copula 
"""

class Stochastic_model():
    def __init__(self, input_names, shared=False, mesh_density=400):
        self.marginals = {}
        for i, name in enumerate(input_names):
            self.marginals[name] = {}
        self.Ndimensions = len(input_names)
        self.label = ''
        self.shared = shared
        self.mesh_density = mesh_density

    def generate_marginal(self, **kwargs):
        for key, value in kwargs.items():
            if self.shared:
                marginal=value
            else:
                marginal = PDF(value)
            marginal.label = key
            self.marginals[key] = marginal
            setattr(self, key, marginal)

    def generate_correlation_matrix(self, partial_correlations):

        self.correlation_matrix = correlation_matrix(
            self.Ndimensions, partial_correlations)

    def choose_copula(self, theta, copula_choice='Gaussian'):
        print('TODO : impliment other copulas!!!')
        """
        Implimented all the bivariate archemedian copulas from the Wiki Page:

        https://en.wikipedia.org/wiki/Copula_(probability_theory)
        """

        self.copula_choice = copula_choice
        copula_families = ['Gaussian', 'Ali_Mik_Haq',
                           'Clayton', 'Frank', 'Gumbel', 'Joe']
        if self.copula_choice not in copula_families:
            print('{} not in implimented bivariate copulas implimented: \n\t choose from: {}'.format(
                self.copula_choice, copula_families))
        #elif self.copula_choice == 'Gaussian':
        #    print('Using multivariate Gauss. Specify a corrlation matrix')

        else:
            if self.copula_choice == copula_families[0]:
                COPULA = Gaussian(theta)
            if self.copula_choice == copula_families[1]:
                if not np.any(theta > -1 and theta < 1):
                    print(r'Error to use Ali–Mikhail–Haq $\theta \in -1,1$')
                COPULA = Ali_Mik_Haq(theta)
            elif self.copula_choice == copula_families[2]:
                if not np.any(theta > -1 and theta != 0):
                    error = r'Error to use Clayton $\theta \in -1,\inf and \neq 0$'
                    print(error)
                COPULA = Clayton(theta)
            elif self.copula_choice == copula_families[3]:
                if not np.any(theta != 0):
                    error = r'Error to use Frank $\theta \neq 0$'
                    print(error)
                COPULA = Frank(theta)
            elif self.copula_choice == copula_families[4]:
                if not np.any(theta > 1):
                    error = r'Error to use Gumbel $\theta \neq 0$'
                    print(error)
                COPULA = Gumbel(theta)
            elif self.copula_choice == copula_families[5]:
                if not np.any(theta > 1):
                    error = r'Error to use Joe $\theta \neq 0$'
                    print(error)
                COPULA = Joe(theta)

            self.COPULA = COPULA
            #super().__init__(self.copula1)

    def sample(self, Nsamples):

        if self.copula_choice == 'Gaussian':
            if not hasattr(self, 'correlation_matrix'):
                print('No correlation defined! Assuming independence')
                parcor = np.zeros(len(self.marginals.keys()))
                self.generate_correlation_matrix(parcor)
            # sample from the gaussian copula
            c = cholesky(self.correlation_matrix)
            # TODO : Call copula from choice of copulas
            mvnorm = stats.multivariate_normal([0, 0], c)
            x = mvnorm.rvs(Nsamples)
            #z = np.random.normal(size=[Nsamples, self.Ndimensions])
            #x = np.matrix(z) * np.matrix(c).T       # project correlation
            #x = np.array(x)
            # Normalise samples
            x = stats.norm.cdf(x)
        else:
            if not hasattr(self, '_pcdf'):
                self.compute_cdf()

            #if self.mesh_density < Nsamples:
            #    print(
            #        'WARNING: Sample size is lower than CDF mesh density, increasing mesh density!')
            #    self.mesh_density = 2*Nsamples
            #    self.mesh_sample()
            #    print('Reevaluating CDF')
            #    self.compute_cdf()

            U = np.random.uniform(0, 1, Nsamples)
            x = self.conditional_sample(U)

        S = np.zeros([Nsamples, self.Ndimensions])
        for i, name in enumerate(self.marginals.keys()):
            S[:, i] = getattr(self, name).inverse_transform(x[:, i])

        return S

    def conditional_sample(self, X):
        Y = np.zeros(len(X))
        pY = np.linspace(0, 1, len(self._x_0))

        for i in range(len(X)):
            i_y = np.argmin(abs(self._x_0-X[i]))
            ppf_y0 = self._pcdf[i_y, :]

            if i_y == len(self._x_0)-1:
                ppf_y1 = self._pcdf[i_y-1, :]
            else:
                ppf_y1 = self._pcdf[i_y+1, :]

            ppf_y = (ppf_y1-ppf_y0)/(self._x_0[1]-self._x_0[0])
            y0 = np.random.uniform(0, 1, 1)
            Y[i] = np.interp(y0, ppf_y, pY)

        self._samples = np.concatenate([[X], [Y]]).T
        return self._samples

    def Like(self, X, V):
        """ 
        Likelihood of X from v
        """
        px = np.linspace(0, 1, len(self._x_0))
        u = np.argmin(abs(self._x_0-X))
        v = np.argmin(abs(self._x_0-V))
        return self._pdf[u, v]

    def LogL(self,x,v):
        L = []
        for i in range(len(x)):
            L.append(self.Like(x[i],v[i]))

        return np.sum(np.log(np.array(L)))

    def H(self, W, X, inv=True):
        """
        Conditioning at X and evaulating the inverse at W
        X \in [0,1]
        W \in [0,1]
        """
        #Y = np.zeros(len(X))
        pY = np.linspace(0, 1, len(self._x_0))

        #for i in range(len(X)):
        i_y = np.argmin(abs(self._x_0-X))
        ppf_y0 = self._pcdf[i_y, :]

        if i_y == len(self._x_0)-1:
            ppf_y1 = self._pcdf[i_y-1, :]
        else:
            ppf_y1 = self._pcdf[i_y+1, :]

        ppf_y = (ppf_y1-ppf_y0)/(self._x_0[1]-self._x_0[0])

        if inv:
            Y = np.interp(W, ppf_y, pY)
        else:
            Y = np.interp(W, pY, ppf_y)
        return Y

    def H_mat(self,W,X,inv=True):
        
        h_out = []
        for i in range(len(X)):
            h_out.append(self.H(W[i],X[i],inv=inv))
        return np.mean(h_out)

    def mesh_sample(self):
        #TODO : impliment differential mesh density to optionally
        # modify the accuracy of interpolation on the different dimensions
        # useful if one of the pdfs is dirty
        
        self._x_0 = np.linspace(0, 1, self.mesh_density)
        self._um, self._vm = np.meshgrid(self._x_0, self._x_0)
        self.U = np.vstack(self._um)
        self.V = np.vstack(self._vm)

    def compute_cdf(self):
        if not hasattr(self, '_us'):
            self.mesh_sample()
        self._pcdf = self.COPULA.copula(self.U, self.V)

    def compute_pdf(self):
        if not hasattr(self, '_pcdf'):
            self.compute_cdf()
        pdf = np.zeros(np.shape(self._pcdf))
        pdf[1:-1, 1:-1] = np.gradient(np.gradient(self._pcdf[1:-1, 1:-1],
                                                  self._x_0[1:-1], axis=1), self._x_0[1:-1], axis=0)
        pdf[1:, 0] = pdf[1:,1]
        pdf[0, 1:] = pdf[1, 1:]
        pdf[0,0] =pdf[-1,-1] = np.max(pdf[1:-1,1:-1])
        self._pdf = pdf/(len(self._x_0))


    def partial_deriv(self, df_w_0=[]):
        print('\t Computing partial deriverative of copula {}'.format(
            self.label))
        dfu = np.gradient(self._pcdf, self._x_0, axis=0)
        dfv = np.gradient(self._pcdf, self._x_0, axis=1)
        self.df_u = dfu
        self.df_v = dfv
        self.df_uv = dfu*dfv

    def plot_copula_cdf(self, cmap='hsv', n_levels=10):
        #if self.copula_choice == 'Gaussian':
        #print('TODO : impliment for gaussian')
        #else:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        contours = plt.contour(self._um, self._vm, self._pcdf, cmap=cmap)
        plt.clabel(contours, inline=True, fontsize=n_levels)

        plt.xlabel(list(self.marginals.keys())[0])
        plt.ylabel(list(self.marginals.keys())[1])
        ax.set_aspect('equal', adjustable='box')
        plt.show()
    
    def plot_copula_pdf(self,cmap='hsv',Nsamples=5000,levels=400):
        marg_lab = list(self.marginals.keys())[0]
        Px = self.marginals[marg_lab].ppf
        Fx = self.marginals[marg_lab].inverse_transform

        SAMP_X = Px(self.sample(Nsamples))
        xy = np.vstack([SAMP_X[:, 0], SAMP_X[:, 1]])
        z = gaussian_kde(xy)(xy)

        plt.scatter(SAMP_X[:, 0], SAMP_X[:, 1], c=z, marker='+')

        labs = plt.contour(self._um, self._vm, self._pdf,
                           levels=levels, alpha=0.9, cmap=cmap)
        plt.clabel(labs, inline=True, fontsize=10)
        plt.show()

    def keys(self):
        return self.__dict__.keys()

"""
Class with some methods for numerical calculus on bivariate copula 
"""
class Copula_calculus():
    def __init__(self, copula, name=[]):
        print('\n__init__: Copula calculus object')
        self._tmp_copula = copula
        if name:
            self._label_copula(name)

    def _label_copula(self, name):
        self.name_1 = name

    def lagged_copula(self, copula_0):
        print('\t Computing conditional partial deriverative of copula {} - copula {}'.format(
            copula_0.label, self._tmp_copula.label))
        # self._um, self._vm
        #print('\t Computing {}'.format(r'$C_{t,t2} = \intg_{[0,1]} C_{0}(u,w)/dw$ C_{0}(u,w)/dw$ C_{t,t+1}(w,v)/dw'))
        dwdt = np.zeros(np.shape(self._tmp_copula.df_u))

        size = self._tmp_copula._x_0.shape[0]
        cell = self._tmp_copula._x_0

        dU = copula_0.df_u
        dV = self._tmp_copula.df_v

        self._tmp_copula.dU = dU
        self._tmp_copula.dV = dV
        W_interp = np.linspace(1,0, size)
        for i in range(size):
            ind_U = np.argmin(abs(cell-W_interp[i]))
            for j in range(size):
                ind_V = np.argmin(abs(cell-W_interp[j]))

                dw = abs(1/size * dU[:, ind_V] * dV[ind_U, :])
                dwdt[i, j] = np.sum(dw)

        self._tmp_copula.unconditional_pcdf = self._tmp_copula._pcdf
        self._tmp_copula._pcdf = dwdt

        setattr(self, self.name_1, copy.deepcopy(self._tmp_copula))
        return self.name_1, self._tmp_copula

    def keys(self):
        return self.__dict__.keys()

"""
Class to develop an N dimensional multivariate distribution with a D Vine copula
"""

class D_stochastic_model(Copula_calculus):
    def __init__(self, stoch_model0, stoch_model1, _level=1):

        self.stoch_model0 = copy.deepcopy(stoch_model0)
        self.stoch_model1 = copy.deepcopy(stoch_model1)
        self._level = _level
        if not hasattr(self.stoch_model0, 'df_u'):
            print('Missing df_u computing partial deriv of self.stoch_model0')
            self.stoch_model0.partial_deriv()
        if not hasattr(self.stoch_model1, 'df_u'):
            print('Missing df_u computing partial deriv of self.stoch_model1')
            self.stoch_model1.partial_deriv()

    def compute_conditional_copula(self):

        self._mod0_vars = list(self.stoch_model0.marginals.keys())
        self._mod1_vars = list(self.stoch_model1.marginals.keys())

        if self._level < 3:
            self.sharred_variable = list(
                set(self._mod0_vars) & set(self._mod1_vars))[0]
            v0 = list(self.stoch_model0.marginals.keys())
            v1 = list(self.stoch_model1.marginals.keys())
            v0.remove(self.sharred_variable)
            v1.remove(self.sharred_variable)

        else:  # self._level == 2 :
            A0, A1 = self._mod0_vars[1].split('_')
            B0, B1 = self._mod1_vars[1].split('_')
            self.sharred_variable = '{}{}_{}{}'.format(A0, B1, B0, A1)
            v0 = '{}{}'.format(A0, B1)
            v1 = '{}{}'.format(B0, A1)
        #else:
        #    print('ERROR_NOTHING PROVIDED')

        Var0 = self.sharred_variable
        Var1 = '{}_{}'.format(v0[0], v1[0])

        Cop_Name = 'Dcop_{}_{}{}_{}'.format(
            self._level, v0[0], v1[0], self.sharred_variable)
        super().__init__(self.stoch_model0)
        self._label_copula(Cop_Name)
        copN, d_vine_cop = self.lagged_copula(self.stoch_model1)

        # Udating marginals of the conditional copula
        marginals = {}
        marg_den = PDF(np.random.uniform(
            0, 1, self.stoch_model0.mesh_density))
        marginals[Var0] = marg_den
        marginals[Var1] = marg_den
        marginals[Var0].label = Var0
        marginals[Var1].label = Var1
        setattr(d_vine_cop, Var0, marg_den)
        setattr(d_vine_cop, Var1, marg_den)
        d_vine_cop.marginals = marginals
        return copN, d_vine_cop

"""
Class containing methods to use the D vine copula 
"""
class D_vine():
    def __init__(self, marginal, theta, copula_choice, stationary=True):
        print('Initialising D vine')
        self.marginal = marginal
        self.theta = theta
        self.copula_choice = copula_choice

        self._copula_node_array = []
        self.stationary = stationary
        if stationary:
            self._stationary_marginal()

        self.mesh_density = 800
    
    def _stationary_marginal(self):
        self.marginal = PDF(self.marginal)
        print('Evaluating Kernel Density once to save time!')
        t0 = time.time()
        self.marginal.fit_kde()
        t1 = time.time()
        print('KDE evaluation complete: Time Elapsed : {}'.format(t1-t0))

    def fit_first_order(self):
        self.order_1_copula = fit_first_order_copula(
            self.marginal, self.theta, self.copula_choice, stationary=self.stationary, mesh_density=self.mesh_density)

        self._copula_node_array.append(
            list(getattr(self, 'order_1_copula').keys()))

        pdf = getattr(self.order_1_copula[self._copula_node_array[0][0]],
                      self.order_1_copula[self._copula_node_array[0][0]].marginal_labels[0])

        print('Setting stationary marginal :\n\t ppf : self.Px \n\t Fx : self.Fx')
        self.Px = pdf.ppf
        self.Fx = pdf.inverse_transform
        self.PDF = pdf

    def fit_conditional_copula_row(self, Theta_Mat,level=2):
        """
        # Just impliment serial dependence and the see how it is to simulate
        Vals = getattr(self, 'order_{}_copula'.format(level-1))
        N_copula = len(Vals)
        Keys = list(Vals)
        tmp = fit_higher_order_copula(Vals, Keys, level=level)
        """
        
        N_lag = Theta_Mat.shape[0]
        prev_N_lag = len(list(getattr(self, 'order_{}_copula'.format(level-1)).keys()))
        # check Layer length:
        if not prev_N_lag - 1 == N_lag:
            print('ERROR: Theta_Mat length must be one less that lower layer \n\t Theta_Mat.shape[0] = {} - Must be {}'.format(N_lag,prev_N_lag -1))
            return
        #Theta_Mat = np.concatenate([[np.nan], Theta_Mat])
        print('Setting Uniform Marginals.')
        Marg = PDF(np.random.uniform(0, 1, 1000))

        setattr(self,'order_{}_copula'.format(level),{})
        #for j in range(1, N_lag): 
        #    Marg.label = 'I{}_{}'.format(j,j+1)
        setattr(self, 'order_{}_copula'.format(level),fit_first_order_copula(Marg, Theta_Mat, self.copula_choice, stationary=True, mesh_density=self.mesh_density))
        self._copula_node_array.append(
            list(getattr(self, 'order_{}_copula'.format(level)).keys()))
            
    def fit_nth_order(self):

        level = 2
        Vals = getattr(self, 'order_{}_copula'.format(level-1))
        #return Vals
        N_copula = len(self.order_1_copula)
        while N_copula > 1:
            Keys = list(Vals)
            tmp = fit_higher_order_copula(Vals, Keys, level=level)
            setattr(self, 'order_{}_copula'.format(level), tmp)
            N_copula = len(getattr(self, 'order_{}_copula'.format(level)))

            self._copula_node_array.append(
                list(getattr(self, 'order_{}_copula'.format(level)).keys()))

            level += 1
            Vals = getattr(self, 'order_{}_copula'.format(level-1))

    def predict_D_vine_3(self, X=[]):
        var = 3
        if len(X):
            w = np.concatenate([X, np.random.uniform(0, 1, 1)])
            x = np.concatenate([X, [np.array(0)]])
            if np.any(np.array(x)) == 1:
                x = np.random.uniform(0,1,3)
        else:
            w = np.random.uniform(0, 1, var)
            x = np.zeros(np.shape(w))
            x[0] = w[0]
            x[1] = self.order_1_copula[1].H(w[1], x[0], inv=True)

        A = self.order_1_copula[1].H(x[1], x[0], inv=False)
        B = self.order_1_copula[2].H(w[2], A, inv=True)
        x[2] = self.order_2_copula[1].H(B, x[0], inv=True)
        return x

    def _samp_vine(self, X=[]):
        """
        ===================================================================
        K. Aas, C. Czado, A. Frigessi, and H. Bakken. 
        Pair-copula constructions of multiple dependence.Insurance:  
        Mathematics and Economics, 44(2):182–198, 2009
        ===================================================================
        Algorithm 2 Simulation algorithm for D-vine. Generates one sample x1, . . . , xn from the vine. Sample
        """
        copula_array = deserialise_copula(self)
        
        n_var = len(getattr(self, '_copula_node_array')[0])+1
        Arr = getattr(self, '_copula_node_array')
        lev = 1
        j = 1

        if len(X):
            w = np.concatenate([[np.array(np.nan)],X, np.random.uniform(0, 1, 1)])
        else:
            w = np.concatenate([[np.array(np.nan)], np.random.uniform(0, 1, n_var)])

        #print('W:{}'.format(w))

        x = np.zeros(np.shape(w))
        v = np.zeros([np.shape(w)[0]+n_var, np.shape(w)[0]+n_var])

        x[1] = v[1, 1] = w[1]
        #getattr(self, 'order_{}_copula'.format(lev))[j]
        x[2] = v[2, 1] = copula_array[1,1].H(w[2], v[1, 1], inv=True)
        #getattr(self, 'order_{}_copula'.format(lev))[j]
        v[2, 2] = copula_array[1, 1].H(v[1, 1], v[2, 1], inv=False)

        for i in inclusive_range(3, n_var,1):

            #print('V:{}'.format(v))
            #print('I {}'.format(i))

            # one main for-loop containing one for- loop for sampling the variables
            v[i, 1] = w[i]
            for k in range(i-1, 2, -1):
                #getattr(self, 'order_{}_copula'.format(i-k))[k]
                v[i, 1] = copula_array[i -k,k].H(v[i, 1], v[i-1, 2*k-2], inv=True)
            # getattr(self, 'order_{}_copula'.format(i-1))[1]
            v[i, 1] = copula_array[i-1,1].H(v[i, 1], v[i-1, 1], inv=True)
            x[i] = v[i, 1]
            if i == n_var:
                break
            #getattr(self, 'order_{}_copula'.format(i-1))[1]
            v[i, 2] = copula_array[i-1,1].H(v[i-1, 1], v[i, 1], inv=False)
            #getattr(self, 'order_{}_copula'.format(i-1))[1]
            v[i, 3] = copula_array[i-1,1].H(v[i, 1], v[i-1, 1], inv=False)

            if i > 3:
                for j in range(2, i-2,1):
                    #one for-loop for computing the needed conditional distribution functions
                    #getattr(self, 'order_{}_copula'.format(i-j))[j]
                    v[i, 2*j] = copula_array[i-j,j].H(v[i-1, 2*j-2], v[i, 2*j-1], inv=False)
                    #getattr(self, 'order_{}_copula'.format(i-j))[j]
                    v[i, 2*j+1] = copula_array[i-j,j].H(v[i, 2*j-1], v[i-1, 2*j-2], inv=False)

            #print('A:{},{}'.format(i, 2*i-2))
            #print('CopLev:{}'.format(i-1))
            #print('CopI:{}'.format(1))
            #print('V0:{},{}'.format(i-1, 2*i-4))
            #print('V1:{},{}'.format(i, 2*i-3))
            #getattr(self, 'order_{}_copula'.format(1))[i-1]
            
            v[i, 2*i-2] = copula_array[i -
                                       1,1].H(v[i-1, 2*i-4], v[i, 2*i-3], inv=False)
        return x[1:]

    def loglikelihood(self,X):
        """
        ===================================================================
        K. Aas, C. Czado, A. Frigessi, and H. Bakken.
        Pair-copula constructions of multiple dependence.Insurance:
        Mathematics and Economics, 44(2):182–198, 2009
        ===================================================================
        Algorithm 4 Likelihood evaluation for D-vine. 

        ==============================
        X must be indexed from 1 !!!!!
        ==============================
        """
        copula_array = deserialise_copula(self)
        
        n_var = len(getattr(self, '_copula_node_array')[0])+1
        Arr = getattr(self, '_copula_node_array')

        logL = 0 
        V = np.zeros([n_var+1,n_var+1])
        V0 = []
        V0.append(np.nan)
        for i in range(1,n_var+1):
            V0.append(X[i])

        for i in range(1, n_var-1):
            logL = logL + copula_array[1, i].Like(V0[i], V0[i+1])
        
        V[1, 1] = copula_array[1, 1].H(V0[1], V0[2], inv=False)

        for k in range(1, n_var-3):
            V[1, k] = copula_array[1, k+1].H(V0[k+2],V0[k+1],inv=False)
            V[1, 2*k+1] = copula_array[1, k +
                                       1].H(V0[k+1], V0[k+2], inv=False)
        print('V:{}'.format([1, 2*n_var-4]))
        print('cop:{}'.format([1, n_var - 1]))
        print('nVar:{}'.format([1, n_var - 1]))

        V[1, 2*n_var-4] = copula_array[1, n_var -1].H(V0[n_var], V0[n_var-1], inv=False)

        for j in range(2, n_var-1):
            for i in range(1, n_var-j):
                logL = logL + copula_array[j,
                                           i].Like(V[j-1, 2*i-1], V[j-1, 2*i])
            if j == n_var-1:
                break
            V[j, 1] = copula_array[j, 1].H(V[j-1, 1], V[j-1, 2], inv=False)
            if n_var>4:
                for i in range(1,n_var-j-2):
                    V[j,2*i] = copula_array[j,i+1].H(V[j-1,2*i+2], V[j-1,2*i+1], inv=False)
                    V[j, 2*i+1] = copula_array[j, i +1].H(V[j-1, 2*i+1], V[j-1, 2*i+2], inv=False)
            V[j, (2*n_var)-(2*j)-2] = copula_array[j, n_var-j].H(V[j -1, 2*n_var-2*j], V[j-1, (2*n_var)-(2*j)-1], inv=False)
        return logL



    def forcast_vine(self, X=[]):
        return self._samp_vine(self, X=X)

    def sample_vine(self,Nsamples):
        n_var = len(getattr(self, '_copula_node_array')[0])+1
        Y = np.zeros([Nsamples,n_var])

        for i in range(Nsamples):
            Y[i, :] = self._samp_vine()
            c = 0
            while np.any(Y[i, :]>0.991):
                Y[i, :] = self._samp_vine()
                c=+1
                if c >30:
                    Y[i, :] =np.ones(n_var)*0.5
        return Y 


    def keys(self):
        return self.__dict__.keys()

    def sampling_constructor(self, X):
        return 'TODO'  # Method takes X, Values at the lags.


def inclusive_range(start, stop, step):
    return range(start, (stop + 1) if step >= 0 else (stop - 1), step)

def deserialise_copula(vine):
    N_ar = vine._copula_node_array[0][-1]+1
    copula_array = np.empty([N_ar, N_ar], dtype=object)
    copula_array[:, 0] = np.nan
    copula_array[0, :] = np.nan
    lev = 0
    for i in range(1, N_ar):
        for j in vine._copula_node_array[i-1]:
            
            copula_array[j+lev, i] = getattr(vine,'order_{}_copula'.format(i))[j]
            copula_array[i,j+lev] = getattr(vine,'order_{}_copula'.format(i))[j]
        lev += 1
    return copula_array


def deserialise_copula_DIAG(vine):
    N_ar = vine._copula_node_array[0][-1]+1
    copula_array = np.empty([N_ar, N_ar], dtype=object)
    copula_array[:, 0] = np.nan
    copula_array[0, :] = np.nan
    lev = 1

    # FILL DIAG
    for j in vine._copula_node_array[0]:
        copula_array[j, j] = vine.order_1_copula[j]

    # Populate the rest
    for i in range(1, N_ar-1):
        for j in vine._copula_node_array[i]:
            copula_array[j+i, j] = getattr(vine,
                                           'order_{}_copula'.format(i+1))[j]

    return copula_array

def fit_first_order_copula(Marginal, THETA, COPULA_CHOICE, stationary=True, mesh_density=800):
    N_lag = len(THETA)
    copula_labels = ['T0_I{}'.format(N_lag-i) for i in range(N_lag)]
    variables = ['L{}'.format(N_lag-i) for i in range(N_lag+1)]

    COP = {}
    if len(COPULA_CHOICE)<2:
        print('Only one copula selected!')
        COPULA_CHOICE = [COPULA_CHOICE]*N_lag

    for i in range(N_lag):
        if stationary:
            Marginal0 = copy.deepcopy(Marginal)
            Marginal1 = copy.deepcopy(Marginal0)
            Marginal0.label = variables[i]
            Marginal1.label = variables[i+1]
            inputs = {variables[i]: Marginal0, variables[i+1]: Marginal1}
            copula = Stochastic_model([variables[i], variables[i+1]],shared=True, mesh_density=mesh_density)
        else:
            copula = Stochastic_model([variables[i], variables[i+1]], mesh_density=mesh_density)
            inputs = {variables[i]: Marginal, variables[i+1]: Marginal}

        copula.generate_marginal(**inputs)
        print('COPULA CHOICE: {}'.format(COPULA_CHOICE[i]))
        copula.choose_copula(THETA[i], copula_choice=COPULA_CHOICE[i])
        copula.mesh_sample()
        print('\tCoputing PDF and CDF')
        copula.compute_cdf()
        copula.compute_pdf()
        copula.label = copula_labels[i]
        copula.marginal_labels = [variables[i], variables[i+1]]
        COP[i+1] = copula
    return COP


def fit_higher_order_copula(copula_dict, copula_list, level=1):
    print('#'*35)
    print('Evaluating Copulae at level : {}'.format(level))

    N_cop = len(copula_list)-1
    cop_list_2 = {}
    c=0
    for i in range(N_cop,0,-1):
        print('='*35+'\n Copula {} and {}'.format(i,i-1))
        print('\tCoputing PDF and CDF')
        copula_dict[copula_list[i]].compute_cdf()
        copula_dict[copula_list[i-1]].compute_cdf()
        copula_dict[copula_list[i]].partial_deriv()
        copula_dict[copula_list[i-1]].partial_deriv()
        copula_dict[copula_list[i]].compute_pdf()
        copula_dict[copula_list[i-1]].compute_pdf() 

        Dvine = D_stochastic_model(copula_dict[copula_list[i-1]],copula_dict[copula_list[i]], _level=level)
        cop_name, cop = Dvine.compute_conditional_copula()
        c+=1
        cop_list_2[c] = {}
        cop_list_2[c] = cop

    print('Completed Evaluation of Copulae at level : {}'.format(level))
    print('#'*35)
    return cop_list_2

"""
Copula sampling class
"""


def correlation_matrix(d, partial_corrs):
    P = np.zeros([d, d])
    S = np.eye(d)

    for k in range(d-1):
        ll = 1
        for i in range(k+1, d):
            P[k, i] = partial_corrs[ll]
            p = P[k, i]
            for l in inclusive_range(k-1, 0, -1):
                p = p * np.sqrt((1-P[l, i]**2) *
                                (1-P[l, k]**2))+(P[l, i]*P[l, k])
            S[k, i] = p
            S[i, k] = p
            ll = ll+1
    return S


def n_p_cor(res):
    k = res-1
    for i in range(k-1, 0, -1):
        k += i
    return int(k)


def inclusive_range(start, stop, step):
    return range(start, (stop + 1) if step >= 0 else (stop - 1), step)

"""
Implimented copula classes 
"""
class Student():
    def __init__(self,theta):
        self.theta = theta

    def copula(self,U,V):
        return print('MISSING')

class Gaussian():
    def __init__(self, theta):
       
        self.theta = theta
    """
    def generator(self, t):
        return -np.log(1-(1-t)**self.theta)

    def inv_generator(self, t):
        return 1-(1-np.exp(-t))**(1/self.theta)
    """

    def correlation_matrix(self):
        self.sigma = np.array([[1, self.theta], [self.theta, 1]])

    def copula(self, U, V):
        A = 1/np.sqrt(1-self.theta**2)
        a = np.sqrt(2) * erfinv(2*U-1)
        b = np.sqrt(2) * erfinv(2*V-1)
        B = np.exp(-((a**2+b**2)*self.theta**2 - 2*a *
                     b*self.theta)/(2*(1-self.theta**2)))

        return A*B


class Ali_Mik_Haq():
    def __init__(self, theta):
        self.theta = theta

    def generator(self, t):
        return np.log((1-self.theta*(1-t))/(t))

    def inv_generator(self, t):
        return (1-self.theta)/(np.exp(t)-self.theta)

    def copula(self, U, V):
        A = (U*V)/(1-(self.theta*(1-U)*(1-V)))
        return A  # np.concatenate([[U], [V]])


class Frank():
    def __init__(self, theta):
        self.theta = theta

    def generator(self, t):
        A = np.exp(-self.theta*t) - 1
        B = np.exp(-self.theta) - 1
        return -np.log(A/B)

    def inv_generator(self, t):
        return -(1/self.theta)*np.log(1+np.exp(-t)*(np.exp(-self.theta)-1))

    def copula(self, U, V):
        A = (np.exp(-self.theta*U)-1)*(np.exp(-self.theta*V)-1)
        B = np.exp(-self.theta)-1
        C = -(1/self.theta)*np.log(1+(A/B))
        return C  # np.concatenate([[U], [V]])


class Gumbel():
    def __init__(self, theta):
        self.theta = theta

    def generator(self, t):
        return (-np.log(t))**self.theta

    def inv_generator(self, t):
        return np.exp(-t**(1/self.theta))

    def copula(self, U, V):
        A = (-np.log(U))**self.theta
        B = (-np.log(V))**self.theta
        return np.exp(-(A+B)**(1/self.theta))


class Joe():
    def __init__(self, theta):
        self.theta = theta

    def generator(self, t):
        return -np.log(1-(1-t)**self.theta)

    def inv_generator(self, t):
        return 1-(1-np.exp(-t))**(1/self.theta)

    def copula(self, U, V):
        return 1 - ((1-U)**self.theta+(1-V)**self.theta-(1-U)**self.theta*(1-V)**self.theta)**(1/self.theta)


class Clayton():
    def __init__(self, theta):
        self.theta = theta

    def generator(self, t):
        return (1/self.theta) * (t**(-self.theta)-1)

    def inv_generator(self, t):
        return (1 + self.theta*t)**(-1/self.theta)

    def copula(self, U, V):
        A = U**(-self.theta) + V**(-self.theta)-1
        try:
            A[A < 0] = 0
        except:
            A = np.max([A, 0])
        A = A**(-1/self.theta)
        return A  # np.concatenate([[U], [V]])

    def pdf(self, U, V):
        A = self.theta+1 * (U*V)**(-(self.theta+1))
        B = (U**(-self.theta) + V**(-self.theta) -
             1)**(-(2*self.theta + 1)/self.theta)
        return A*B

    def ppf(self, Y, V):
        A = Y**(self.theta/(-1-self.theta))
        B = Y**self.theta
        return ((A + B - 1)/B)**(-1/self.theta)



"""
Miscilaneous stuff
"""


def ecdf(x):
    xs = np.sort(x)
    #xs = np.append(xs,xs[-1])
    n = xs.size
    y = np.linspace(0, 1, n)
    #np.arange(1, n+1) / n
    #xs = np.append(xs[0],xs)
    #ps =
    return [y, xs]

def confidence_interval_mean(data, confidence_interval=99, plot=True):
    " This is only appropriate for one d data array"

    data = data.squeeze()
    n = np.shape(data)
    if len(n) > 1:
        print('ERROR: Data array must be 1 dimensional- dimensions = {}'.format(len(n)))

    n = n[0]
    x_bar = np.mean(data)
    s = np.std(data)
    t_alpha = stats.t.ppf(1 - ((100-confidence_interval)/2/100), n-1)

    l_alpha = x_bar - t_alpha * (s/np.sqrt(n))
    h_alpha = x_bar + t_alpha * (s/np.sqrt(n))

    if plot:
        py, d = ecdf(data)
        plt.step(d, py, c='blue', label='Data')
        plt.plot(np.ones(2)*h_alpha, [0, 1], c='red', alpha=90,
                 label='{} mean ci '.format(1 - ((100-confidence_interval)/2/100)))
        plt.plot(np.ones(2)*l_alpha, [0, 1], c='red', alpha=70,
                 label='{} mean ci '.format(((100-confidence_interval)/2/100)))
        plt.title('sample mean {} % ci'.format(confidence_interval))
        plt.legend()
    return [l_alpha, h_alpha]


def s_ln(x, data):
    n = np.size(data)
    l = np.sum(data <= x)
    return l/n


def smirnov_critical_value(alpha, n):
    # a = np.array([0.20,0.15,0.10,0.05,0.025,0.01,0.005,0.001])
    # c_a = np.array([1.073,1.138,1.224,1.358,1.48,1.628,1.731,1.949])
    #
    # if any(np.isclose(0.0049,a,2e-2)):
    # c_alpha = c_a[np.where(np.isclose(0.0049,a,2e-2))[0]]
    # else:
    c_alpha = np.sqrt(-np.log(alpha/2)*(1/2))
    return (1/np.sqrt(n))*c_alpha


def confidence_limits_distribution(x, alpha, interval=False, n_interp=100, plot=False, x_lim=[-10, 10], label=''):
    """
    The confidence limits of F(x) is an inversion of the well known KS-test.
    KS test is usually used to test whether a given F(x) is the underlying probability distribution of Fn(x).

    See      : Experimental uncertainty estimation and statistics for data having interval uncertainty. Ferson et al.
               for this implimentation. Here interval valued array is valid.
    """

    if not interval:
        data = np.zeros([2, np.size(x)])
        data[0] = x
        data[1] = x
    else:
        data = x

    x_i = np.linspace(np.min(data[0])+x_lim[0],
                      np.max(data[1])+x_lim[1], n_interp)

    N = np.size(data[0])

    if N < 50:
        print('Dont trust me! I really struggle with small sample sizes\n')
        print('TO DO: Impliment the statistical conversion table for Z score with lower sample size')

    def b_l(x): return min(
        1, s_ln(x, data[0])+smirnov_critical_value(round((1-alpha)/2, 3), N))

    def b_r(x): return max(
        0, s_ln(x, data[1])-smirnov_critical_value(round((1-alpha)/2, 3), N))

    L = []
    R = []
    for i, xi in enumerate(x_i):
        L.append(b_l(xi))
        R.append(b_r(xi))

    if plot:
        pl, xl = ecdf(data[0])
        pr, xr = ecdf(data[1])
        plt.step(xl, pl, color='blue', label='data', alpha=0.3)
        plt.step(xr, pr, color='blue', alpha=0.7)
        plt.step(x_i, L, color='red', label='data', alpha=0.7)
        plt.step(x_i, R, color='red', alpha=0.7,
                 label='KS confidence limits {}%'.format(alpha))
        plt.xlabel(label)
        plt.ylabel('P(x)')
    return L, R, x_i


def compute_tau_kendall_lags(Data, Lags):

    tau = []
    p_value = []
    for i in Lags:
        ta = stats.kendalltau(Data[:-i], Data[i:])
        tau.append(ta[0])
        p_value.append(ta[1])
    return tau, p_value

def plot_lag_scatter(Data, lag, label, save_address=[],rows=2,cmap='jet'):
    if isinstance(Data,list):
        flag= False
    else:
        flag= True
    
    a = int(rows)
    b = int(len(lag)/2)
    fig, (axs) = plt.subplots(ncols=b, nrows=a,
                              figsize=(b*5, a*5), sharex='col', sharey='row')
    fig.subplots_adjust(hspace=0.2, wspace=0.03)
    c = 0
    axs[0, 0].set_ylabel(label)
    for f, i in enumerate(lag):
        f0 = f
        if f > b-1:
            f0 = f - b

        if f == b:
            c += 1
            axs[c, 0].set_ylabel(label)
        if flag:
            xy = np.vstack([Data[:-i], Data[i:]])
        else:
            xy = np.vstack([Data[f][0], Data[f][1]])
        z = gaussian_kde(xy)(xy)
        
        if flag:
            axs[c, f0].scatter(Data[:-i], Data[i:], c=z, marker='+')
        else:
            axs[c, f0].scatter(Data[f][0], Data[f][1], c=z, marker='+',cmap=cmap)

        axs[c, f0].text(0.01, 0.9, s='Lag : {}'.format(i),
                        horizontalalignment='left',
                        verticalalignment='top',
                        transform=axs[c, f0].transAxes, size=18, c='red')
        if c == a-1:
            axs[c, f0].set_xlabel(label)
    if save_address:
        plt.savefig(save_address)
    plt.show()
