import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt

from epidemic_data import Epidemic_data
from epidemic_sir import Epidemic_SIR

def main():

    path = ( './COVID-19/csse_covid_19_data/csse_covid_19_time_series/'
         + 'time_series_19-covid-' )

    covid = Epidemic_data(path)

    covid.grasp(type='all', country='Italy', rm_geo=True, rm_zero=True,
        transpose=True)
    data_ita = covid.get_data()

    sir = Epidemic_SIR(data_ita)

if (__name__ == "__main__"):
    main()
