import pandas as pd
import numpy as np

class Epidemic_data(object):

    def __init__(self, path, format):

        # read raw
        confirmed = pd.read_csv(path + 'confirmed_global.csv')
        confirmed['Type'] = 'confirmed'
        deaths = pd.read_csv(path + 'deaths_global.csv')
        deaths['Type'] = 'deaths'
        recovered = pd.read_csv(path + 'recovered_global.csv')
        recovered['Type'] = 'recovered'

        #deaths.set_index('Country/Region', inplace=True)
        #confirmed.set_index('Country/Region', inplace=True)
        #recovered.set_index('Country/Region', inplace=True)

        # dataframe representing all dataset in tabular format
        self.data = pd.concat([confirmed, deaths, recovered], axis=0,
            ignore_index=True)

        self.out = pd.DataFrame()

        self.format = format

    def get_data(self):
        return self.out

    def grasp(self, type='all', country='Italy', rm_geo=True, rm_zero=True,
        transpose=True, format='resolved'):

        # select only country of interest
        self.out = self.data[self.data['Country/Region'] == country]

        # select time series of interest
        if (type == 'confirmed' or type == 'deaths' or type == 'recovered' ):
            self.out = self.out[self.out['Type'] == type]

        # remove geographic informations
        if (rm_geo):
            self.out = self.out.iloc[:,4:]

        if (rm_zero):
            # index of first non zero case
            bool_idx = [self.out.drop('Type', axis=1).sum() > 0]
            idx = np.argmax(bool_idx)
            self.out = self.out.iloc[:,idx:]

        # traspose, set a datetime index and add a column 'time'
        # representing n. of days from start day to support fit

        if (transpose):
            self.out.set_index('Type', inplace=True)
            self.out.index.name = None
            self.out = self.out.transpose()
            self.out.set_index(
                pd.to_datetime(self.out.index, infer_datetime_format=True),
                inplace=True)
            self.out = self.out.asfreq('D')
            self.out['time'] = self.out.index - self.out.index[0]
            self.out['time'] = self.out['time'].dt.days

        if (self.format == 'resolved'):
            self.out['resolved'] = self.out['deaths'] + self.out['recovered']
            self.out = self.out.drop('deaths', axis=1)
            self.out = self.out.drop('recovered', axis=1)

        self.out = self.out.dropna()

    def timespan(self):
        return [np.min(self.out['time']), np.max(self.out['time'])]

    def initial_value(self):
        tmp = self.out[self.out['time'] == 0]
        tmp = tmp.drop('time', axis=1)
        return tmp.values[0]

    def time(self):
        return self.out['time'].values

    def time_series(self):
        return self.out.drop('time', axis=1).values
        #confirmed_glb = confirmed.iloc[:,3:].transpose().sum(axis=1)
        #my_pd['confirmed_glb'] = confirmed_glb.values

        #recovered_glb = recovered.iloc[:,3:].transpose().sum(axis=1)
        #my_pd['recovered_glb'] = recovered_glb.values

        #my_pd['time'] = my_pd.index - my_pd.index[0]
        #my_pd['time'] = my_pd['time'].dt.days
#def extend_index(self, index, new_size):
#    values = index.values
#    current = datetime.strptime(index[-1], '%m/%d/%y')
#    while len(values) < new_size:
#        current = current + timedelta(days=1)
#        values = np.append(values, datetime.strftime(current, '%m/%d/%y'))
#    return values
