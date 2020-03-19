import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt

def main():
    covid = grasp_data(
        './data/csse_covid_19_data/csse_covid_19_time_series/' +
        'time_series_19-covid-')
    #plot_dashboard(covid)

    #fit_glb = my_fit(covid, 'time', 'confirmed_glb', 0.9, [10,1000,300000],
    ##    True)

    #print(fit_glb)

    fit_ita_c = logistic_fit(covid, 'time', 'confirmed_ita', 0.9, [1,10,100000],
        True)

    print(fit_ita_c)

    fit_ita_r = logistic_fit_2(covid, 'time', 'resolved_ita', 0.9, [5,50],
        True)

    print(fit_ita_r)

    fit_ita_p = pulse_fit(covid, 'time', 'resolved_perc', 1, [4.85,32,4.6,20],
        ([4.8,31,4.4,19], [4.9,34,4.6,21]), True)

    print(fit_ita_p)

    plt.figure()
    tmp_data = covid[covid['resolved_perc']>0]
    yy_points = tmp_data['resolved_perc'].values.astype(np.float)
    xx_points = tmp_data['time'].values.astype(np.float)
    xx_points = xx_points - xx_points[0]
    time = np.arange(3 * np.max(xx_points))
    plt.plot(xx_points, yy_points, '.r')
    plt.plot(time, pulse_model(time, fit_ita_c[0][0], fit_ita_c[0][1],
        fit_ita_r[0][0], fit_ita_r[0][1]))

    plt.show()

#1.06354377e+05

def pulse_fit(data, xx_name, yy_name, cross, p0, bounds, plt_flag):
    tmp_data = data[data[yy_name]>0]
    yy_points = tmp_data[yy_name].values.astype(np.float)
    xx_points = tmp_data[xx_name].values.astype(np.float)
    xx_points = xx_points - xx_points[0]

    time = np.arange(3 * np.max(xx_points))

    max_idx = np.ceil(cross * len(xx_points)).astype(np.int)
    print(max_idx)

    fit = opt.curve_fit(pulse_model, xx_points[:max_idx],
        yy_points[:max_idx], p0=p0, bounds=bounds)

    if (plt_flag):
        plt.figure()
        axes = plt.gca()
        plt.subplot(2,1,1)
        plt.plot(xx_points, yy_points, '.r')
        plt.plot(xx_points[:max_idx], yy_points[:max_idx], 'oy')
        plt.plot(time, pulse_model(time, fit[0][0], fit[0][1],
            fit[0][2], fit[0][3]))
        plt.subplot(2,1,2)
        plt.plot(time[1:], np.diff(pulse_model(time, fit[0][0], fit[0][1],
            fit[0][2], fit[0][3])))
        plt.plot(xx_points[1:],np.diff(yy_points), '.r')

    return fit

def logistic_fit_2(data, xx_name, yy_name, cross, p0, plt_flag):

    tmp_data = data[data[yy_name]>0]
    yy_points = tmp_data[yy_name].values.astype(np.float)
    xx_points = tmp_data[xx_name].values.astype(np.float)
    xx_points = xx_points - xx_points[0]

    time = np.arange(3 * np.max(xx_points))

    max_idx = np.ceil(cross * len(xx_points)).astype(np.int)
    print(max_idx)

    fit = opt.curve_fit(logistic_model_2, xx_points[:max_idx],
        yy_points[:max_idx], p0=p0)

    if (plt_flag):
        plt.figure()
        axes = plt.gca()
        plt.subplot(2,1,1)
        plt.plot(xx_points, yy_points, '.r')
        plt.plot(xx_points[:max_idx], yy_points[:max_idx], 'oy')
        plt.plot(time, logistic_model_2(time, fit[0][0], fit[0][1]))
        plt.subplot(2,1,2)
        plt.plot(time[1:], np.diff(logistic_model_2(time, fit[0][0], fit[0][1])))
        plt.plot(xx_points[1:],np.diff(yy_points), '.r')

    return fit


def logistic_fit(data, xx_name, yy_name, cross, p0, plt_flag):

    tmp_data = data[data[yy_name]>0]
    yy_points = tmp_data[yy_name].values.astype(np.float)
    xx_points = tmp_data[xx_name].values.astype(np.float)
    xx_points = xx_points - xx_points[0]

    time = np.arange(3 * np.max(xx_points))

    max_idx = np.ceil(cross * len(xx_points)).astype(np.int)
    print(max_idx)

    fit = opt.curve_fit(logistic_model, xx_points[:max_idx],
        yy_points[:max_idx], p0=p0)

    if (plt_flag):
        plt.figure()
        axes = plt.gca()
        plt.subplot(2,1,1)
        plt.plot(xx_points, yy_points, '.r')
        plt.plot(xx_points[:max_idx], yy_points[:max_idx], 'oy')
        plt.plot(time, logistic_model(time, fit[0][0], fit[0][1],
            fit[0][2]))
        plt.subplot(2,1,2)
        plt.plot(time[1:], np.diff(logistic_model(time, fit[0][0], fit[0][1],
            fit[0][2])))
        plt.plot(xx_points[1:],np.diff(yy_points), '.r')

    return fit

def logistic_model_2(x, a, b):
    return 1.06354377e+05 / (1 + np.exp(-(x-b)/a))

def logistic_model(x, a, b, c):
    return c / (1 + np.exp(-(x-b)/a))

def pulse_model(x, a, b, c, d):
    return (1 + np.exp(-(x-d)/c)) / (1 + np.exp(-(x-b)/a))

def gen_logistic_model(x, a, b, c, d):
    c * (1 + np.exp(-(x-b)/a))**(-d)


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

    my_pd['resolved_ita'] = recovered_ita.values + deaths_ita.values
    my_pd['resolved_perc'] = my_pd['resolved_ita']/my_pd['confirmed_ita']

    deaths_glb = deaths.iloc[:,3:].transpose().sum(axis=1)
    my_pd['deaths_glb'] = deaths_glb.values

    confirmed_glb = confirmed.iloc[:,3:].transpose().sum(axis=1)
    my_pd['confirmed_glb'] = confirmed_glb.values

    recovered_glb = recovered.iloc[:,3:].transpose().sum(axis=1)
    my_pd['recovered_glb'] = recovered_glb.values


    my_pd['time'] = my_pd.index - my_pd.index[0]
    my_pd['time'] = my_pd['time'].dt.days

    return my_pd


if (__name__ == "__main__"):
    main()
