

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
