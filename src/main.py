import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def main():
    covid = grasp_data(
        './data/csse_covid_19_data/csse_covid_19_time_series/' +
        'time_series_19-covid-')
<<<<<<< HEAD
    t = range(len(d))
    print(d.columns[1])
=======
    #t = range(len(d))
    #print(d.loc[d>0,1])
>>>>>>> d9ecbfc72f0c91b46b8091c0fe72b0d9663c57d0
    #np.polyfit(t, np.log(d.loc[]), 1, w=numpy.sqrt(y))

    covid['r_by_c_glb'] = covid['recovered_glb'] / covid['confirmed_glb']
    covid['r_by_c_ita'] = covid['recovered_ita'] / covid['confirmed_ita']

    covid['d_by_r_glb'] = covid['deaths_glb'] / covid['recovered_glb']
    covid['d_by_r_ita'] = covid['deaths_ita'] / covid['recovered_ita']

    plt.figure()
    axes = plt.gca()
    covid.plot(style=".", y=['deaths_ita', 'confirmed_ita', 'recovered_ita'],
        color="red", ax=axes)
    covid.plot(kind="line", y=['deaths_ita', 'confirmed_ita', 'recovered_ita'],
        color="green", ax=axes)
    plt.figure()
    axes = plt.gca()
    covid.plot(style=".", y=['deaths_glb', 'confirmed_glb', 'recovered_glb'],
        color="red", ax=axes)
    covid.plot(kind="line", y=['deaths_glb', 'confirmed_glb', 'recovered_glb'],
        color="blue", ax=axes)
    plt.figure()
    axes = plt.subplot(2,2,1)
    covid.plot(kind="line", y=['r_by_c_glb'], color="black", ax=axes)
    axes = plt.subplot(2,2,2)
    covid.plot(kind="line", y=['r_by_c_ita'], color="green", ax=axes)
    #plt.figure()
    #axes = plt.gca()
    axes = plt.subplot(2,2,3)
    covid.plot(kind="line", y=['d_by_r_glb'], color="black", ax=axes)
    axes = plt.subplot(2,2,4)
    covid.plot(kind="line", y=['d_by_r_ita'], color="green", ax=axes)
    plt.show()

def grasp_data(path):

    deaths = pd.read_csv(path + 'Deaths.csv')
    confirmed = pd.read_csv(path + 'Confirmed.csv')
    recovered = pd.read_csv(path + 'Recovered.csv')

    deaths.set_index('Country/Region', inplace=True)
    confirmed.set_index('Country/Region', inplace=True)
    recovered.set_index('Country/Region', inplace=True)

    deaths_ita = deaths.loc['Italy'].iloc[3:].transpose()
    my_pd = pd.DataFrame(index=pd.to_datetime(deaths_ita.index))
    my_pd['deaths_ita'] = deaths_ita.values

    confirmed_ita = confirmed.loc['Italy'].iloc[3:].transpose()
    my_pd['confirmed_ita'] = confirmed_ita.values

    recovered_ita = recovered.loc['Italy'].iloc[3:].transpose()
    my_pd['recovered_ita'] = recovered_ita.values

    deaths_glb = deaths.iloc[:,3:].transpose().sum(axis=1)
    my_pd['deaths_glb'] = deaths_glb.values

    confirmed_glb = confirmed.iloc[:,3:].transpose().sum(axis=1)
    my_pd['confirmed_glb'] = confirmed_glb.values

    recovered_glb = recovered.iloc[:,3:].transpose().sum(axis=1)
    my_pd['recovered_glb'] = recovered_glb.values

    return my_pd


if (__name__ == "__main__"):
    main()
