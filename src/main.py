import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def main():
    d, c, r = grasp_data(
        './data/csse_covid_19_data/csse_covid_19_time_series/' +
        'time_series_19-covid-')
    t = range(len(d))
    print(d.columns[1])
    #np.polyfit(t, np.log(d.loc[]), 1, w=numpy.sqrt(y))
    d.plot(style=".", color="red")
    d.plot(color="blue")
    plt.show()

def grasp_data(path):
    deaths = pd.read_csv(path + 'Deaths.csv')
    confirmed = pd.read_csv(path + 'Confirmed.csv')
    recovered = pd.read_csv(path + 'Recovered.csv')

    deaths.set_index('Country/Region', inplace=True)
    confirmed.set_index('Country/Region', inplace=True)
    recovered.set_index('Country/Region', inplace=True)

    deaths_ita = deaths.loc['Italy'].iloc[3:].transpose()
    deaths_ita.index = pd.to_datetime(deaths_ita.index)

    confirmed_ita = confirmed.loc['Italy'].iloc[3:].transpose()
    confirmed_ita.index = pd.to_datetime(confirmed_ita.index)

    recovered_ita = recovered.loc['Italy'].iloc[3:].transpose()
    recovered_ita.index = pd.to_datetime(recovered_ita.index)

    return deaths_ita, confirmed_ita, recovered_ita


if (__name__ == "__main__"):
    main()
