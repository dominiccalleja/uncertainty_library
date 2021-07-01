from scipy.linalg import eigh, cholesky
from sklearn.neighbors import KernelDensity

import sys 

import numpy as np
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

import copy
import seaborn as sns
import matplotlib.gridspec as gridspec
mpl.rcParams.update({'font.size': 10})

import scipy.signal as sig
import scipy.stats as stats
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

    def describe_object(self):
        self._axis_label = '{} ({})'.format(self.label, self.units)

    def inverse_transform(self, X):
        return np.interp(X, self.P, self.X)

    def ppf(self, X):
        return np.interp(X, self.X, self.P)

    def inverse_transform_kde(self, N):
        if not hasattr(self, '_kde'):
            self.fit_kde()
        return np.interp(X, self.P, self.X)

    def sample(self, N):
        U = np.random.uniform(0, 1, N)
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
        self._kde = KernelDensity(
            kernel='gaussian', bandwidth=0.75).fit(self.data)
        return self._kde

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

    def plot_confidence_distribution(self, alpha=0.95, n_interp=1000, xlim=[-1, 1]):
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
Copula sampling class
"""


class Copula_sample():
    def __init__(self, copula):
        self.copula = copula

    def mesh_sample(self, mesh_density=1000):
        #TODO : impliment differential mesh density to optionally
        # modify the accuracy of interpolation on the different dimensions
        # useful if one of the pdfs is dirty
        self._mesh_density = mesh_density
        self._x_0 = np.linspace(0, 1, mesh_density)
        self._um, self._vm = np.meshgrid(self._x_0, self._x_0)
        self.U = np.vstack(self._um)
        self.V = np.vstack(self._vm)

    def compute_cdf(self):
        if not hasattr(self, '_us'):
            self.mesh_sample()
        self._pcdf = self.copula.copula(self.U, self.V)

    def conditional_sample(self, Nsamples):
        if not hasattr(self, '_pcdf'):
            self.compute_cdf()

        if self._mesh_density < Nsamples:
            print(
                'WARNING: Sample size is lower than CDF mesh density, increasing mesh density!')
            self.mesh_sample(mesh_density=2*Nsamples)
            print('Reevaluating CDF')
            self.compute_cdf()

        X = np.random.uniform(0, 1, Nsamples)
        Y = np.zeros(Nsamples)
        pY = np.linspace(0, 1, len(self._x_0))

        for i in range(Nsamples):
            i_y = np.argmin(abs(self._x_0-X[i]))
            ppf_y0 = self._pcdf[i_y, :]

            if i_y == len(self._x_0)-1:
                ppf_y1 = self._pcdf[i_y-1, :]
            else:
                ppf_y1 = self._pcdf[i_y+1, :]

            ppf_y = (ppf_y1-ppf_y0)/(self._x_0[1]-self._x_0[0])
            y0 = np.random.uniform(0, 1, 1)
            Y[i] = np.interp(y0, ppf_y, pY)

        self._samples = np.concatenate([[X], [Y]])
        return self._samples

    def plot_copula_cdf(self, cmap='hsv'):
        if self.copula_choice == 'Gaussian':
            print('TODO : impliment for gaussian')
        else:
            contours = plt.contour(self._um, self._vm, self._pcdf, cmap=cmap)
            plt.clabel(contours, inline=True, fontsize=8)
            
            try:
                plt.xlabel(self.copula.marginal_labels.keys()[0])
                plt.ylabel(self.copula.marginal_labels.keys()[1])
            except:
                print('No variable labels.')
            plt.show()
            
            




"""
Class to make stochastic model 
"""


class Stochastic_model(Copula_sample):
    def __init__(self, input_names):
        self.marginals = {}
        for i, name in enumerate(input_names):
            self.marginals[name] = {}
        self.Ndimensions = len(input_names)

        self.label = ''

    def generate_marginal(self, **kwargs):
        for key, value in kwargs.items():
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
        elif self.copula_choice == 'Gaussian':
            print('Using multivariate Gauss. Specify a corrlation matrix')

        else:
            if self.copula_choice == copula_families[1]:
                if not (theta > -1 and theta < 1):
                    print(r'Error to use Ali–Mikhail–Haq $\theta \in -1,1$')
                COPULA = Ali_Mik_Haq(theta)
            elif self.copula_choice == copula_families[2]:
                if not (theta > -1 and theta != 0):
                    error = r'Error to use Clayton $\theta \in -1,\inf and \neq 0$'
                    print(error)
                COPULA = Clayton(theta)
            elif self.copula_choice == copula_families[3]:
                if not (theta != 0):
                    error = r'Error to use Frank $\theta \neq 0$'
                    print(error)
                COPULA = Frank(theta)
            elif self.copula_choice == copula_families[4]:
                if not (theta > 1):
                    error = r'Error to use Gumbel $\theta \neq 0$'
                    print(error)
                COPULA = Gumbel(theta)
            elif self.copula_choice == copula_families[5]:
                if not (theta > 1):
                    error = r'Error to use Joe $\theta \neq 0$'
                    print(error)
                COPULA = Joe(theta)
            self.copula = COPULA
            super().__init__(self.copula)

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
            #self.sapler = Copula_sample(self.copula)
            x = self.conditional_sample(Nsamples).T

        X = np.zeros([Nsamples, self.Ndimensions])
        for i, name in enumerate(self.marginals.keys()):
            X[:, i] = getattr(self, name).inverse_transform(x[:, i])

        return X

    def add_dependent_copula(self, copula2, Ind_Var, Dep_Var):
        print('Adding a third dependent variable with another copula')
        self.shared_variable = Ind_Var
        self.second_dependent_variable = Dep_Var
        self.copula2 = copula2
        if not hasattr(self, '_pcdf'):
            print('Missing CDF for Copula 1. Computing now!')
            self.compute_cdf()
        if not hasattr(self.copula2, '_pcdf'):
            print('Missing CDF for Copula 2. Computing now!')
            self.copula2.compute_cdf()

    def second_order_sample(self, Nsamples):
        if not hasattr(self, 'copula2'):
            print(
                'ERROR: This method requires an additonal copula to be added to this model')
            print('You can use this by creating a new Stochastic_model object with one sharred \n variable and then use add_dependent_copula()')

        if self.shared_variable not in list(self.marginals.keys()):
            print('ERROR: Inependent variable {} is not in this joint distribution! \n\t choose from {}'.format(
                self.shared_variable, list(self.marginals.keys())))
            return
        if self.shared_variable not in list(self.copula2.marginals.keys()):
            print('ERROR: Inependent variable {} is not in this joint distribution! \n\t choose from {}'.format(
                self.shared_variable, list(self.copula2.marginals.keys())))
            return
        if self.second_dependent_variable not in list(self.copula2.marginals.keys()):
            print('ERROR: Dependent variable {} is not in this joint distribution! \n\t choose from {}'.format(
                self.second_dependent_variable, list(self.copula2.marginals.keys())))
            return

        # Sample this object joint distribution
        X = self.sample(Nsamples)
        var_ind = list(self.marginals.keys()).index(self.shared_variable)

        X2 = second_order_sample(
            self, self.copula2, X[:, var_ind], self.shared_variable, self.second_dependent_variable, Nsamples)

        samples = np.concatenate([X.T, [X2]], axis=0).T
        return samples

    def add_dependent_copula_series(self, copula_list, Ind_Var_list, Dep_Var_list):
        print('Adding a list of dependent copulas')
        self.shared_var_list = Ind_Var_list
        self.depend_var_list = Dep_Var_list
        self.copula_list = copula_list

        for i, cop in enumerate(self.copula_list):
            getattr(cop, 'compute_cdf')()

    def construct_D_vine_copula(self):
        print('\n\nComputing the D-Vine copula series.')
        print('WARNING: I\'m doing a lot of numerical integration! I may take a while for very long series!')
        print('Don\'t worry here is a handry progress bar for you to know whats occurin!')
        print('\nTODO: Assuming the list of copulas is from Lag-h to Lag-1. Need to add option for other way around!')

        NLAG = len(self.copula_list)
        print('Evaluating the condintional D-Vine copula for {} Lags'.format(NLAG))
        print('')
        toolbar_width = 100
        # setup toolbar
        sys.stdout.write("[{}]".format(" " * toolbar_width))
        sys.stdout.flush()
        sys.stdout.write("\b" * (1))
        prog_lim = int(100/len(self.copula_list))
        progress_c = 0
        i = 1
        copula_list = self.copula_list

        cop0 = copula_list[0]
        cop_cal0 = Copula_calculus(cop0)
        cop_cal0.partial_deriv()
        d_uw0 = copula_list[0].df_u
        copula_list.pop(0)

        progress_c += i*prog_lim
        sys.stdout.write(
            '\r'+'[{} - {}%'.format(progress_c*'#', progress_c))
        sys.stdout.flush()

        D_Vine_cop = []
        D_Vine_cop.append(self.copula_list[0]._pcdf)
        for cop in copula_list:
            cop_cal1 = Copula_calculus(cop)
            cop_cal1.partial_deriv()
            d_uw1 = cop_cal1.lagged_copula(cop0)
            D_Vine_cop.append(d_uw1)

            cop0 = copy.deepcopy(cop)

            i += 1
            progress_c = i*prog_lim
            sys.stdout.write(
                '\r'+'[{} - {}%'.format(progress_c*'#', progress_c))
            sys.stdout.flush()

        sys.stdout.write("]\n")
        self.D_Vine_cop = D_Vine_cop
        print('Complete. The copula series is now in self.D_Vine_cop')
        print('Overwritten the bivariate copulas CDF with the new conditional D-Vine CDF is self.copula_list._pcdf')

    def plot_D_vine(self, save_address=[]):

        fig, (axs) = plt.subplots(ncols=3, nrows=len(self.copula_list),
                                  sharey=True, sharex=True, figsize=(15, 5*len(self.copula_list)))
        fig.subplots_adjust(hspace=0.1, wspace=0.1)

        for i in range(len(self.copula_list)):
            col = axs[i, 0].contour(self.copula_list[i]._um, self.copula_list[i]._vm,
                                    self.copula_list[i]._pcdf, cmap='jet', levels=30)
            axs[i, 0].clabel(col, inline=True, fontsize=10)
            axs[i, 0].set_ylabel(self.copula_list[i].label)
        axs[i, 0].set_xlabel('Independent Copula CDF')
        for i in range(len(self.copula_list)):
            col = axs[i, 1].contour(self.copula_list[i]._um, self.copula_list[i]._vm,
                                    self.D_Vine_cop[i], cmap='jet', levels=30)
            axs[i, 1].clabel(col, inline=True, fontsize=10)
        axs[i, 1].set_xlabel('Conditional Copulas CDF')
        for i in range(len(self.copula_list)):
            col = axs[i, 2].contour(self.copula_list[i]._um, self.copula_list[i]._vm,
                                    self.copula_list[i]._pcdf-self.D_Vine_cop[i], cmap='Dark2_r', levels=15)
            axs[i, 2].clabel(col, inline=True, fontsize=10)
        axs[i, 2].set_xlabel('Difference CDF')
        if save_address:
            fig.savefig(save_address)
        plt.show()

    def sample_copula_series(self, Nsamples):
        # Sample this object joint distribution
        X = self.sample(Nsamples)
        self.copula_series_samples = []
        self.copula_series_samples.append(X[:, 0])
        self.copula_series_samples.append(X[:, 1])

        var_ind = 1
        X = X[:, var_ind]
        base = copy.deepcopy(self)
        for i, cop in enumerate(self.copula_list):
            X = second_order_sample(
                base, cop, X, self.shared_var_list[i], self.depend_var_list[i], Nsamples)
            base = copy.deepcopy(cop)
            self.copula_series_samples.append(X)
        return self.copula_series_samples

    def keys(self):
        return self.__dict__.keys()


class Copula_calculus():
    def __init__(self, copula_t):
        print('\n__init__: Copula calculus object')
        self.copula_t = copula_t

    def partial_deriv(self, df_w_0=[]):
        print('\t Computing partial deriverative of copula {}'.format(
            self.copula_t.label))
        dfu = np.gradient(self.copula_t._pcdf, self.copula_t._x_0, axis=0)
        dfv = np.gradient(self.copula_t._pcdf, self.copula_t._x_0, axis=1)
        self.copula_t.df_u = dfu
        self.copula_t.df_v = dfv
        self.copula_t.df_uv = dfu*dfv

    def lagged_copula(self, copula_0):
        print('\t Computing conditional partial deriverative of copula {} - copula {}'.format(
            copula_0.label, self.copula_t.label))
        # self._um, self._vm
        #print('\t Computing {}'.format(r'$C_{t,t2} = \intg_{[0,1]} C_{0}(u,w)/dw$ C_{0}(u,w)/dw$ C_{t,t+1}(w,v)/dw'))
        dwdt = np.zeros(np.shape(self.copula_t.df_u))

        size = self.copula_t._x_0.shape[0]
        cell = self.copula_t._x_0

        dU = copula_0.df_u
        dV = self.copula_t.df_v

        self.dU = dU
        self.dV = dV
        W_interp = np.linspace(0, 1, size)
        for i in range(size):
            ind_U = np.argmin(abs(cell-W_interp[i]))
            for j in range(size):
                ind_V = np.argmin(abs(cell-W_interp[j]))

                dw = 1/size * dU[:, ind_U] * dV[ind_V, :]
                dwdt[i, j] = np.sum(dw)

        self.copula_t.unconditional_pcdf = self.copula_t._pcdf
        self.copula_t._pcdf = dwdt
        lag_copula = dwdt
        return lag_copula

def second_order_sample(obj1, obj2, values, Ivar, Dvar, Nsamples):

    # getting P values of independent variable
    Px = getattr(obj1, Ivar).ppf(values).T

    pY = np.linspace(0, 1, len(obj2._x_0))
    Y = np.zeros(Nsamples)
    obj2.mesh_sample()  # generating new copula mesh for copula2
    for i in range(Nsamples):
        # finding cdf of dependent variable
        i_y = np.argmin(abs(obj2._x_0-Px[i]))
        ppf_y0 = obj2._pcdf[i_y, :]  # extracting cdf of dependent variable
        if i_y == len(obj2._x_0)-1:
            ppf_y1 = obj2._pcdf[i_y-1, :]
        else:
            ppf_y1 = obj2._pcdf[i_y+1, :]

        # partial deriverative of dependent variable
        ppf_y = (ppf_y1-ppf_y0)/(obj2._x_0[1]-obj2._x_0[0])
        y0 = np.random.uniform(0, 1, 1)              # uniform sample
        # inverse transform dependent CDF
        Y[i] = np.interp(y0, ppf_y, pY)

    Y = getattr(obj2, Dvar).inverse_transform(Y)
    return Y


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
