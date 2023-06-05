from matplotlib import use as m_use
m_use('TkAgg')

from bcb import sgs
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import math
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.filters.hp_filter import hpfilter
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf
from scipy.stats import jarque_bera
from scipy.stats import probplot
import pylab

data = sgs.get({'GDP' : 4380, # Seas adjusted GDP
                'Infl' : 189, # IGP-M
                 'MS' : 27841, # Money Supply
                 'i' : 4390 # Nominal interest rate
                 }, start = '2000-01-01', end = '2023-04-01')

data = pd.DataFrame(data)

data = data.dropna()

# Deflating

data['ms'] = data['MS']/np.cumprod(data['Infl'] /100 + 1)
adfuller(np.log(data['ms'])) # Not stationary :/
data['ms_1'] = data['ms'].shift(1) # Using lagged value to diminish autocorrelation

data['gdp'] = data['GDP']/np.cumprod(data['Infl'] /100 + 1)
adfuller(np.log(data['gdp'])) # Not stationary :/

data = data.dropna()

# Hodrick–Prescott filtering on log values
y = np.array(hpfilter(np.log(data['ms']), lamb = 1600)[0])

X = np.array([hpfilter(np.log(data['gdp']), lamb = 1600)[0],
              hpfilter(np.log(data['i']), lamb = 1600)[0],
              hpfilter(np.log(data['ms_1']), lamb = 1600)[0]])
X = X.transpose()

model = LinearRegression(fit_intercept=True)
model.fit(X, y)
model.coef_ # Elasticities
e_gdp, e_i = model.coef_[:-1]
model.score(X, y) #R2 values

print('Elasticity for GDP: %.3f\n'\
      'Elasticity for i: %.3f\n' % (e_gdp, e_i))

residuals = y - model.predict(X)

sns.lineplot(y = residuals, x = range(0, len(residuals)))
acf(residuals, nlags = 10)

# Checking for normality

sns.distplot(x = residuals)
probplot(residuals, dist = 'norm', plot = pylab)

stat, p = jarque_bera(residuals)

print('---------------------------------\n'\
    'Jarque-Bera test for normality\n'\
    '---------------------------------\n'\
    'Jarque-Bera: %.3f   P-value: %.3f\n' % (stat, p))
if p >= 0.05 :
        print('Evidence of normality\n')
    else :
        print('No evidence of normality\n')
print('---------------------------------')

# Trend
data['Trend'] = range(1, len(data)+ 1)
data['Trend2'] = data['Trend']^2
y = np.array(np.log(data['ms']))

X = np.array([np.log(data['gdp']),
              np.log(data['i']),
              data['Trend'],
              data['Trend2']])

X = X.transpose()

model = LinearRegression(fit_intercept=True)
model.fit(X, y)
model.coef_ # Elasticities
e_gdp, e_i = model.coef_[:-2]
model.score(X, y) #R2 values

print('Elasticity for GDP: %.3f\n'\
      'Elasticity for i: %.3f\n' % (e_gdp, e_i))

residuals = y - model.predict(X)

sns.lineplot(y = residuals, x = range(0, len(residuals)))
acf(residuals, nlags = 10)

# Evidences of autocorrelation. Estimates are not credible

# Checking for normality

sns.distplot(x = residuals)
probplot(residuals, dist = 'norm', plot = pylab)

stat, p = jarque_bera(residuals)

print('---------------------------------\n'\
    'Jarque-Bera test for normality\n'\
    '---------------------------------\n'\
    'Jarque-Bera: %.3f   P-value: %.3f\n' % (stat, p))
if p >= 0.05 :
        print('Evidence of normality\n')
    else :
        print('No evidence of normality\n')
print('---------------------------------')