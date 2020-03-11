import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt

def main():
    covid = grasp_data(
        './data/csse_covid_19_data/csse_covid_19_time_series/' +
        'time_series_19-covid-')

    #plot_dashboard(covid)
    covid['time'] = covid.index - covid.index[0]
    covid['time'] = covid['time'].dt.days

    fit_glb = opt.curve_fit(logistic_model,
        covid['time'].values.astype(np.float),
        covid['recovered_glb'].values.astype(np.float) / 7.74247069e+4)

    print(fit_glb)

    covid_ita = covid[covid['recovered_ita']>0]
    xx = covid_ita['time'].values.astype(np.float)
    xx = xx - xx[0]
    yy = covid_ita['recovered_ita'].values.astype(np.float) / 771.54859429
    fit_ita = opt.curve_fit(logistic_model, xx, yy)

    print(fit_ita)

    axes = plt.gca()
    plt.plot(covid['time'].values,
        covid['recovered_glb'].values.astype(np.float) / 7.74247069e+4, '.r')
    plt.plot(covid['time'].values, logistic_model(covid['time'].values,
        fit_glb[0][0], fit_glb[0][1], fit_glb[0][2]))

    plt.figure()
    axes = plt.gca()
    plt.plot(xx, yy, '.r')
    plt.plot(xx, logistic_model(xx,
        fit_ita[0][0], fit_ita[0][1], fit_ita[0][2]))

    plt.show()

def logistic_model(x, a, b, c):
    return c / (1 + np.exp(-(x-b)/a))

def plot_dashboard(covid):
    covid['r_by_c_glb'] = covid['recovered_glb'] / covid['confirmed_glb']
    covid['r_by_c_ita'] = covid['recovered_ita'] / covid['confirmed_ita']

    covid['d_by_r_glb'] = covid['deaths_glb'] / covid['confirmed_glb']
    covid['d_by_r_ita'] = covid['deaths_ita'] / covid['confirmed_ita']

    plt.figure()
    axes = plt.gca()
    covid.plot(style=".", y=['deaths_ita', 'confirmed_ita', 'recovered_ita'],
        color="red", ax=axes)
    covid.plot(kind="line", y=['deaths_ita', 'confirmed_ita', 'recovered_ita'],
        color="green", ax=axes)
    plt.figure()
    axes = plt.gca()
    covid.plot(style=".", y=['deaths_glb',  'confirmed_glb', 'recovered_glb'],
        color="red", ax=axes)
    covid.plot(kind="line", y=['deaths_glb',  'confirmed_glb', 'recovered_glb'],
        color="blue", ax=axes)
    plt.figure()
    axes = plt.gca()
    covid.diff().plot(style=".", y=[
        #'deaths_glb',
        #'confirmed_glb',
        'recovered_glb'
    ], color="red", ax=axes)

    covid.diff().plot(kind="line", y=[
        #'deaths_glb',
        #'confirmed_glb',
        'recovered_glb'
        ], color="blue", ax=axes)

    plt.figure()
    axes = plt.gca()

    covid.diff().plot(style=".", y=[
        #'deaths_ita',
        #'confirmed_ita',
        'recovered_ita'
        ], color="red", ax=axes)

    covid.diff().plot(kind="line", y=[
        #'deaths_ita',
        #'confirmed_ita',
        'recovered_ita'
        ], color="blue", ax=axes)

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
