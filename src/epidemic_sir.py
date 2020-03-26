import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from datetime import timedelta, datetime

START_DATE = {
  'Japan': '1/22/20',
  'Italy': '1/31/20',
  'Republic of Korea': '1/22/20',
  'Iran (Islamic Republic of)': '2/19/20'
}

class Epidemic_SIR(object):

    def __init__(self, data=None, params=None):
        self.data = data
        print(self.data)
        #self.timespan = [min time, max time]
        #self.iv = [] #set to measured data
        #self.params = params


    #def load_confirmed(self, country):
    #  """
    #  Load confirmed cases downloaded from HDX
    #  """
    #  df = pd.read_csv('data/time_series_19-covid-Confirmed.csv')
    #  country_df = df[df['Country/Region'] == country]
    #  return country_df.iloc[0].loc[START_DATE[country]:]

    #def extend_index(self, index, new_size):
    #    values = index.values
    #    current = datetime.strptime(index[-1], '%m/%d/%y')
    #    while len(values) < new_size:
    #        current = current + timedelta(days=1)
    #        values = np.append(values, datetime.strftime(current, '%m/%d/%y'))
    #    return values

    def model(t, y):
        # Vectorial ODE

        #  Unmarshall state
        S = y[0]
        I = y[1]
        R = y[2]

        # Compute derivatives
        dS = -beta * S * I
        dI = beta * S * I - gamma * I
        dR = gamma * I

        return [dS, dI, dR]

    def observales(y):
        #z = f(y)
        #return z
        return 0

    def loss(point, data):
        # l2 loss

        # model canot be used directly, I should use a lambda f
        # solve initial value problem
        #solution = solve_ivp(self.model, self.timespan, self.iv,
        #    t_eval=self.data['time'], vectorized=True)

        # compute l2 loss
        # TODO: use libary function to compute l2
        #l2_loss = (self.observables(solution.y) - data) ** 2

        #return l2_loss
        return 0

    def predict(self, n_range):

        #seld.data
        #def SIR(t, y):
        #    S = y[0]
        #    I = y[1]
        #    R = y[2]
        #    return [-beta*S*I, beta*S*I-gamma*I, gamma*I]
        #extended_actual = np.concatenate((data.values, [None] * (size - len(data.values))))
        #return new_index, extended_actual, solve_ivp(SIR, [0, size], [S_0,I_0,R_0], t_eval=np.arange(0, size, 1))
        return 0

    def view ():
        #new_index, extended_actual, prediction = self.predict(beta, gamma, data)
        #df = pd.DataFrame({
    #        'Actual': extended_actual,
        #    'S': prediction.y[0],
        #    'I': prediction.y[1],
        #    'R': prediction.y[2]
        #}, index=new_index)
        #fig, ax = plt.subplots(figsize=(15, 10))
        #ax.set_title(self.country)
        #df.plot(ax=ax)
        #fig.savefig(f"{self.country}.png")

        return 0

    def estimate(self):

        #data = self.load_confirmed(self.country)
        #optimal = minimize(
        #    loss,
        #    [0.001, 0.001],
        #    args=(data),
        #    method='L-BFGS-B',
        #    bounds=[(0.00000001, 0.4), (0.00000001, 0.4)]
        #)
        #beta, gamma = optimal.x
        return 0

    def data():
        return 0
