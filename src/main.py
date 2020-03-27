import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt

from epidemic_data import Epidemic_data
from epidemic_sir import Epidemic_SIR

def main():

    path = ( './COVID-19/csse_covid_19_data/csse_covid_19_time_series/'
         + 'time_series_covid19_' )

    covid = Epidemic_data(path, format='resolved')

    covid.grasp(type='all', country='Italy', rm_geo=True, rm_zero=True,
        transpose=True)

    sir = Epidemic_SIR(covid, start_pop=250000)
    sir.estimate()
    sir.predict(150)
    sir.view()

if (__name__ == "__main__"):
    main()
