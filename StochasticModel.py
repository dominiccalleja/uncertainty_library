from sklearn.neighbors import KernelDensity
import numpy as np
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

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
Class to make stochastic model 
"""

class Stochastic_model():
    def __init__(self, input_names):
        self.marginals = {}
        for i, name in enumerate(input_names):
            self.marginals[name] = {}
        self.Ndimensions = len(input_names)

    def generate_marginal(self, **kwargs):
        for key, value in kwargs.items():
            marginal = stok.PDF(value)
            marginal.label = key
            self.marginals[key] = marginal

    def generate_correlation_matrix(self, partial_correlations):
        self.correlation_matrix = correlation_matrix(
            self.Ndimensions, partial_correlations)

    def choose_copula(self, theta, copula_choice='Gaussian'):
        print('TODO : impliment other copulas!!!')
        self.copula_choice = copula_choice
        copula_families = ['Gaussian', 'Ali_Mik_Haq',
                           'Clayton', 'Frank', 'Gumbel', 'Joe']
        if self.copula_choice not in copula_families:
            print('{} not in implimented bivariate copulas implimented: \n\t choose from: {}'.format(
                self.copula_choice, copula_families))
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

    def sample(self, Nsamples):

        if self.copula_choice == 'Gaussian':
            if not hasattr(self, 'correlation_matrix'):
                print('No correlation defined! Assuming independence')
                parcor = np.zeros(len(self.marginals.keys()))
                self.generate_correlation_matrix(parcor)
            # sample from the gaussian copula
            c = up.cholesky(self.correlation_matrix)
            # TODO : Call copula from choice of copulas
            z = np.random.normal(size=[Nsamples, self.Ndimensions])
            x = np.matrix(z) * np.matrix(c).T       # project correlation
            x = np.array(x)
            # Normalise samples
            x = (x-np.min(x, axis=0))/(np.max(x, axis=0)-np.min(x, axis=0))
        else:
            self.sapler = Copula_sample(self.copula)
            x = self.sapler.conditional_sample(Nsamples)

        X = np.zeros([Nsamples, self.Ndimensions])
        for i, name in enumerate(self.marginals.keys()):
            X[:, i] = self.marginals[name].inverse_transform(x[i])
        return X


def correlation_matrix(d, partial_corrs):
    P = np.zeros([d, d])
    S = np.eye(d)

    for k in range(d-1):
        ll = 1
        for i in range(k+1, d):
            P[k, i] = partial_corrs[ll]
            p = P[k, i]
            for l in up.inclusive_range(k-1, 0, -1):
                p = p * np.sqrt((1-P[l, i]**2) *
                                (1-P[l, k]**2))+(P[l, i]*P[l, k])
            S[k, i] = p
            S[i, k] = p
            ll = ll+1
    return S

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
        A = (U*V)/(1-(theta*(1-U)*(1-V)))
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
        contours = plt.contour(self._um, self._vm, self._pcdf, cmap=cmap)
        plt.clabel(contours, inline=True, fontsize=8)
        plt.show()



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
