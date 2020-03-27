import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from datetime import timedelta, datetime



class Epidemic_SIR(object):

    def __init__(self, data=None, params=None, start_pop=200000):
        self.data = data.get_data()
        self.pred = pd.DataFrame()
        self.y_meas = data.time_series()
        self.N = start_pop
        self.timespan = data.timespan()
        self.iv = data.initial_value()
        self.iv = np.hstack([self.N-np.sum(self.iv), self.iv])
        self.time = data.time()

        self.mdl = self.model()
        self.loss = self.loss()
        self.params = params

        self.rms = 0.

    def model(self):
        # Vectorial ODE
        def sir_mdl(theta, t, y, N) :

            #  Unmarshall state
            (S, I, R) = y

            # Unmarshall parameters
            (l, mu, b, g)  = theta

            # Compute derivatives
            dS = l - mu * S - b * I * S / N
            dI = b * I * S / N - (g + mu) * I
            dR = g * I - mu * R

            return [dS, dI, dR]

        mdl = lambda theta, t, y, N : sir_mdl(theta, t, y, N)
        return mdl

    def observables(self, y):
        return y[:,1:3]

    def loss(self):
        # l2 loss

        def l2_loss(theta):

            x = lambda t, y: self.mdl(theta, t, y, self.N)

            # model canot be used directly, I should use a lambda f
            # solve initial value problem
            solution = solve_ivp(x, self.timespan, self.iv,
                t_eval=self.time, vectorized=True)

            # compute l2 loss
            # TODO: use libary function to compute l2
            l2_loss = np.sum(
                (self.observables(solution.y.T) - self.y_meas) ** 2)

            return l2_loss

        # useless
        my_loss = lambda theta: l2_loss(theta)

        return my_loss

    def predict(self, n_range):

        self.pred['time'] = np.arange(n_range).astype(np.int)

        self.pred.set_index(self.data.index[0] +
            self.data.index.freq * self.pred['time'], inplace=True)

        mdl = lambda t, y: self.mdl(self.params, t, y, self.N)
        prediction = solve_ivp(mdl, [0, n_range], self.iv,
            t_eval=self.pred['time'].values, vectorized=True)
        self.pred['susceptible'] = prediction.y[0]
        self.pred['infectious'] = prediction.y[1]
        self.pred['resolved'] = prediction.y[2]

        print(self.pred)

    def view (self):
        plt.figure()
        axes = plt.gca()
        self.data.plot(style=".", y=['confirmed', 'resolved'],
            color=['red', 'magenta'], ax=axes)
        self.pred.plot(kind="line", y=['infectious', 'resolved'],
            color=['yellow', 'blue'], ax=axes)
        plt.title('Cumulated Cases. rms=' + str(self.rms) +
            ". Max=" + str(self.pred.index[self.pred['infectious'].argmax()]))
        #x=[]
        #self.pred.plot.bar( style='o', color='k', y=['infectious'])

        plt.figure()
        axes = plt.gca()
        self.data.diff().plot(style=".", y=['confirmed', 'resolved'],
            color=['red', 'magenta'], ax=axes)
        self.pred.diff().plot(kind="line", y=['infectious', 'resolved'],
            color=['yellow', 'blue'], ax=axes)
        plt.title('Daily Increments')

        plt.figure()
        axes = plt.gca()
        self.data['p'] = self.data['resolved'] / (self.data['confirmed'] +
            self.data['resolved'])
        self.pred['p'] = self.pred['resolved'] / (self.pred['infectious'] +
            self.pred['resolved'])
        self.data.plot(style=".", y=['p'],
            color="red", ax=axes)
        self.pred.plot(kind="line", y=['p'],
            color="yellow", ax=axes)
        plt.title('Resolved Probability')

        plt.show()

    def estimate(self):

        optimal = minimize(
            self.loss,
            [0.001, 1, 0.001, 0.001],
            method='L-BFGS-B',
            bounds=[(1e-8, 5e-1), (1e-8, 2e-1), (1e-8, 5e-1),
            (1e-8, 5e-1)],
            options={'gtol': 1e-9, 'disp': True}
        )
        self.params = optimal.x
        self.rms = np.sqrt(optimal.fun)
        print(self.params)
        print(self.rms)
