import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import seaborn as sns
import math
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import statistics
from scipy.stats import chi2
from sklearn.model_selection import train_test_split
from statsmodels.tsa.seasonal import STL
import statsmodels.tsa.holtwinters as ets
import statistics as stat
from numpy import linalg


#Dummy vars so i can run only pieces of the code at a time for trouble shooting
run_descriptive = 1
run_base = 1
run_arima = 1
run_regression = 1

### Part 1 - Read in Data ###

df = pd.read_csv('/Users/annoy/Documents/Time Series/Final Project/NASDAQCOM.csv')
#print(df.head())
df2 = pd.read_csv('/Users/annoy/Documents/Time Series/Final Project/DCOILWTICO.csv')
#print(df2.head())
df3 = pd.read_csv('/Users/annoy/Documents/Time Series/Final Project/DGS10.csv')
#print(df3.head())
df4 = pd.read_csv('/Users/annoy/Documents/Time Series/Final Project/DTWEXM.csv')
#print(df4.head())
df5 = pd.read_csv('/Users/annoy/Documents/Time Series/Final Project/USD1MTD156N.csv')
#print(df5.head())

df['NASDAQCOM'] = df['NASDAQCOM'].replace(['.'], np.nan).astype(float)
df2['DCOILWTICO'] = df2['DCOILWTICO'].replace(['.'], np.nan).astype(float)
df3['DGS10'] = df3['DGS10'].replace(['.'], np.nan).astype(float)
df4['DTWEXM'] = df4['DTWEXM'].replace(['.'], np.nan).astype(float)
df5['USD1MTD156N'] = df5['USD1MTD156N'].replace(['.'], np.nan).astype(float)

df = pd.merge(df, df2, how='inner', on='DATE')
df = pd.merge(df, df3, how='inner', on='DATE')
df = pd.merge(df, df4, how='inner', on='DATE')
df = pd.merge(df, df5, how='inner', on='DATE')
df['Spread'] = df3['DGS10'] - df5['USD1MTD156N'] #spread between us gov't 10 year and LIBOR
print(df.head())
print(df.describe())
df_no_null = df.interpolate()
print(df_no_null.describe())

series = df_no_null['NASDAQCOM']

if run_descriptive == 1:
    means = []
    variances = []

    for i in range(len(series)):
        temp = series[:i]
        means.append(np.mean(temp))
        if i > 1:
            variances.append(statistics.variance(temp))
        else:
            variances.append(0)

    plt.figure()
    plt.subplot(211)
    plt.plot(means)
    plt.title('Rolling Mean - Preprocessing')
    plt.subplot(212)
    plt.plot(variances)
    plt.title('Rolling Variance - Preprocessing')
    plt.show()

    lags = 50

    plt.figure()
    plt.subplot(211)
    plot_acf(series, ax=plt.gca(), lags=lags)
    plt.subplot(212)
    plot_pacf(series, ax=plt.gca(), lags=lags)
    plt.show()

    plt.figure(figsize=(9, 6))
    sns.heatmap(df.corr(), vmin=-1, vmax=1, annot=True, cmap='Spectral')
    plt.show()

    pd_data = pd.Series(np.array(series),index=pd.date_range('1986-01-02', freq='D', periods=len(series), name='daily nasdaq closing'))

    STL = STL(pd_data)
    res = STL.fit()

    T = res.trend
    S = res.seasonal
    R = res.resid

    fig = res.plot()
    plt.show()

    var_ratio = stat.variance(R) / stat.variance(R + T)
    str_trend = max(0, 1 - var_ratio)
    print(f'The strength of trend for this data set is: {str_trend}')

    var_ratio = stat.variance(R) / stat.variance(R + S)
    str_season = max(0, 1 - var_ratio)
    print(f'The strength of seasonality for this data set is: {str_season}')

### Part 2 - Train/Test Split ###

series_train, series_test = train_test_split(series, shuffle= False, test_size=0.2)
frame_train, frame_test = train_test_split(df_no_null, shuffle= False, test_size=0.2)

### Part 3 - Make Stationary ###

def ADF_Cal(x):
    result = adfuller(x)
    print("ADF Statistic: %f" %result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))

ADF_Cal(series)  # P-value of 1, so definitly time variant

def difference(dataset, interval=1):
    diff = []
    for i in range(interval, len(dataset)):
        value = float(dataset[i] - dataset[i - interval])
        diff.append(value)
    return diff

df_log = np.log(series_train)
print(f"ADF for log of base series is:")
ADF_Cal(df_log)#P-value of -.63, so still time variant

df_log2 = np.log(df_log)
print(f"ADF for log of difference is:")
ADF_Cal(df_log2)#P-value of -.95

diff_1 = np.diff(df_log2)
print(f"ADF for difference of logs is:")
ADF_Cal(diff_1)#P-Value of 0, but variance increases with time, another log transform?

if run_descriptive == 1:
    means = []
    variances = []

    for i in range(len(diff_1)):
        temp = diff_1[:i]
        means.append(np.mean(temp))
        if i > 1:
            variances.append(statistics.variance(temp))
        else:
            variances.append(0)

    plt.figure()
    plt.subplot(211)
    plt.plot(means)
    plt.title('Rolling Mean - Stationary')
    plt.subplot(212)
    plt.plot(variances)
    plt.title('Rolling Variance - Stationary')
    plt.show()

    lags = 50

    plt.figure()
    plt.subplot(211)
    plot_acf(diff_1, ax=plt.gca(), lags=lags)
    plt.subplot(212)
    plot_pacf(diff_1, ax=plt.gca(), lags=lags)
    plt.show()

### Part 4 - Base Models ###
def avg_forecast(train, num):
    tot = 0
    pred = []
    pred_test = []
    train_size = len(train)

    for int in range(1, num+1, 1):

        if int == 1:
            pred.append(0)
        elif int <= train_size:
            pred.append(tot / (int - 1))
        else:
            pred_test.append(tot / (train_size))

        if int <= train_size:
            tot += train[int - 1]

    return pred_test

def naive_forecast(train, num):
    pred = []
    pred_test = []
    train_size = len(train)

    for int in range(1, num+1, 1):

        if int == 1:
            pred.append(0)
        elif int <= train_size:
            pred.append(train[int - 2])
        else:
            pred_test.append(train[train_size - 1])

    return pred_test

def drift_forecast(train, num):
    pred = []
    pred_test = []
    train_size = len(train)

    for int in range(1, num+1, 1):

        if int <= 2:
            pred.append(0)
        elif int <= train_size:
            slope = (train[int - 2] - train[0]) / (int - 2)
            intercept = train[int - 2] - slope * (int - 1)
            calc = slope * (int) + intercept
            pred.append(calc)
            # print(f"At step {int} slope is {slope} and intercept is {intercept} so calculated value in {calc}")
        else:
            slope = (train[train_size - 1] - train[0]) / (train_size - 1)
            intercept = train[train_size - 1] - slope * (train_size)
            calc = slope * (int) + intercept
            pred_test.append(calc)

    return pred_test

def ses_forecast(train, num):
    pred = []
    pred_test = []
    train_size = len(train)
    alpha = .5

    for int in range(1, num+1, 1):

        if int == 1:
            pred.append(0)
            prior_pred = train[int]
        elif int <= train_size:
            calc = alpha * train[int - 2] + (1 - alpha) * prior_pred
            pred.append(calc)
            prior_pred = calc
        else:
            calc = alpha * train[train_size - 1] + (1 - alpha) * prior_pred
            pred_test.append(calc)

    return pred_test


if run_base == 1:
    length = len(series)
    avg_pred = avg_forecast(series_train, length)
    avg_pred = pd.DataFrame(avg_pred).set_index(series_test.index)
    naive_pred = naive_forecast(series_train, length)
    naive_pred = pd.DataFrame(naive_pred).set_index(series_test.index)
    drift_pred = drift_forecast(series_train, length)
    drift_pred = pd.DataFrame(drift_pred).set_index(series_test.index)
    ses_pred = ses_forecast(series_train, length)
    ses_pred = pd.DataFrame(ses_pred).set_index(series_test.index)

    season_train = pd_data = pd.Series(np.array(series_train),index=pd.date_range('1986-01-02', freq='D', periods=len(series_train), name='daily nasdaq closing'))
    holtz_linear = ets.ExponentialSmoothing(season_train, trend='additive', damped=True, seasonal=None).fit()
    holtz_linear_pred = holtz_linear.forecast(steps=len(series_test))
    holtz_linear_pred = pd.DataFrame(holtz_linear_pred).set_index(series_test.index)
    holtz_winter = ets.ExponentialSmoothing(season_train, trend='multiplicative', damped=False, seasonal='additive').fit()
    holtz_winter_pred = holtz_winter.forecast(steps=len(series_test))
    holtz_winter_pred = pd.DataFrame(holtz_winter_pred).set_index(series_test.index)


    plt.figure()
    plt.plot(series_train, label = 'Training Set')
    plt.plot(series_test, label = 'Test Set')
    plt.plot(avg_pred, label = 'Average Forecasting')
    plt.plot(naive_pred, label = 'Naive Forecasting')
    plt.plot(drift_pred, label = 'Drift Forecasting')
    plt.plot(ses_pred, label = 'Simple Exponential Smoothing Forecasting')
    plt.title('Base Models')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(series_train, label = 'Training Set')
    plt.plot(series_test, label = 'Test Set')
    plt.plot(holtz_linear_pred, label = "Holt's Linear Forecasting")
    plt.plot(holtz_winter_pred, label = "Holt's Winter Forecasting")
    plt.title('Base Models')
    plt.legend()
    plt.show()

### Part 5 - ARMA/ARIMA/SARIMA ###
def sigma(autovar, j, k):
    num = []
    for x in range(k):
        temp = []
        for y in range(k):
            if x == k-1:
                temp.append(autovar[j+y+1])
            else:
                itr = abs(j-x+y)
                temp.append(autovar[itr])
        num.append(temp)
    num_trans = np.array(num).T.tolist()

    denom = []
    for x in range(k):
        temp = []
        for y in range(k):
            itr = abs(j - x + y)
            temp.append(autovar[itr])
        denom.append(temp)
    denom_trans = np.array(denom).T.tolist()

    num_det = np.linalg.det(num_trans)
    denom_det = np.linalg.det(denom_trans)
    if abs(denom_det) <= .00001:
        if denom_det < 0:
            denom_det = -.00001
        else:
            denom_det = .00001
    result = num_det/denom_det
    return result

def gpac_table(process, num_row, num_col):
    auto_cor = sm.tsa.stattools.acf(process, nlags = 40)

    #build matrix by calling sigma function
    matrix = []
    for x in range(num_row):
        temp = []
        for y in range(num_col):
            temp.append(sigma(auto_cor, x, y+1))
        matrix.append(temp)
    return matrix

def poles_and_roots(theta, ar_order, ma_order):
    if ar_order > 0:
        ar_coefs = theta[:ar_order]
    else:
        ar_coefs = [0]

    if ma_order > 0:
        ma_coefs = theta[-ma_order:]
    else:
        ma_coefs = [0]

    ar_coefs = np.insert(ar_coefs, 0, 1)
    ma_coefs = np.insert(ma_coefs, 0, 1)

    poles = np.roots(ar_coefs)
    zeros = np.roots(ma_coefs)

    print(f"Coefs are: {ar_coefs} The poles of the process are {poles}")
    print(f"Coefs are: {ma_coefs} The zeros of the process are {zeros}")

if run_arima == 1:
    series_train_np = series_train.to_numpy()
    gpac = gpac_table(series_train, 8, 8)
    sns.heatmap(gpac, vmin=-1, vmax=1, annot=True, cmap='Spectral')
    plt.xticks(np.arange(len(gpac)), np.arange(1, len(gpac) + 1))
    plt.show()

    gpac = gpac_table(diff_1, 8, 8)
    sns.heatmap(gpac, vmin=-1, vmax=1, annot=True, cmap='Spectral')
    plt.xticks(np.arange(len(gpac)), np.arange(1, len(gpac) + 1))
    plt.show()

    na = 1
    nb = 0

    model = sm.tsa.ARMA(series_train, (na, nb)).fit(trend='nc', disp=0)

    covar = model.cov_params()
    covar = covar.to_numpy()

    print('ARIMA Estimate 1')
    for i in range(na):
        cov = covar[i][i]
        cov_range = 1.96 * cov ** (1 / 2)
        coef_low = model.params[i] - cov_range
        coef_high = model.params[i] + cov_range
        print(
            f"AR coeficient of order {i + 1} is estimated to be {model.params[i]} with a confidence interval of {coef_low} to {coef_high}")
    for i in range(nb):
        cov = covar[na + i][na + i]
        cov_range = 1.96 * cov ** (1 / 2)
        coef_low = model.params[i] - cov_range
        coef_high = model.params[i] + cov_range
        print(
            f"MA coeficient of order {i + 1} is estimated to be {model.params[na + i]} with a confidence interval of {coef_low} to {coef_high}")

    pred = []

    for i in range(len(series_test)):
        if i == 0:
            temp = model.params[0] * (-1) * series_train_np[-1]
        elif i == 1:
            temp = model.params[0] * (-1) * pred[0]
        else:
            temp = model.params[0] * (-1) * pred[i - 1]
        pred.append(temp)

    lm1_pred = pd.DataFrame(pred).set_index(series_test.index)

    poles_and_roots(model.params.to_numpy(), 1, 0)

    na = 3
    nb = 0

    model = sm.tsa.ARMA(series_train, (na, nb)).fit(trend='nc', disp=0)

    covar = model.cov_params()
    covar = covar.to_numpy()

    print('ARIMA Estimate 2')
    for i in range(na):
        cov = covar[i][i]
        cov_range = 1.96 * cov ** (1 / 2)
        coef_low = model.params[i] - cov_range
        coef_high = model.params[i] + cov_range
        print(
            f"AR coeficient of order {i + 1} is estimated to be {model.params[i]} with a confidence interval of {coef_low} to {coef_high}")
    for i in range(nb):
        cov = covar[na + i][na + i]
        cov_range = 1.96 * cov ** (1 / 2)
        coef_low = model.params[i] - cov_range
        coef_high = model.params[i] + cov_range
        print(
            f"MA coeficient of order {i + 1} is estimated to be {model.params[na + i]} with a confidence interval of {coef_low} to {coef_high}")

    pred = []

    for i in range(len(series_test)):
        if i == 0:
            temp = model.params[0] * (-1) * series_train_np[-1] + model.params[1] * (-1) * series_train_np[-2] + model.params[2] * (-1) * series_train_np[-3]
        elif i == 1:
            temp = model.params[0] * (-1) * pred[0] + model.params[1] * (-1) * series_train_np[-1] + model.params[2] * (-1) * series_train_np[-2]
        elif i == 2:
            temp = model.params[0] * (-1) * pred[1] + model.params[1] * (-1) * pred[0] + model.params[2] * (-1) * series_train_np[-1]
        elif i == 3:
            temp = model.params[0] * (-1) * pred[2] + model.params[1] * (-1) * pred[1] + model.params[2] * (-1) * pred[0]
        else:
            temp = model.params[0] * (-1) * pred[i - 1] + model.params[1] * (-1) * pred[i - 2] + model.params[2] * (-1) * pred[i - 3]
        pred.append(temp)

    lm2_pred = pd.DataFrame(pred).set_index(series_test.index)

    poles_and_roots(model.params.to_numpy(), 3, 0)

    plt.figure()
    plt.plot(series_train, label='Training Set')
    plt.plot(series_test, label='Test Set')
    plt.plot(lm1_pred, label='First (Order 1, 0) ARIMA Model')
    #plt.plot(lm2_pred, label='Second (Order 3, 0) ARIMA Model')
    plt.title('ARIMA Models')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(series_train, label='Training Set')
    plt.plot(series_test, label='Test Set')
    #plt.plot(lm1_pred, label='First (Order 1, 0) ARIMA Model')
    plt.plot(lm2_pred, label='Second (Order 3, 0) ARIMA Model')
    plt.title('ARIMA Models')
    plt.legend()
    plt.show()



    ### Differenced ARIMA Model
    na = 2
    nb = 2

    model = sm.tsa.ARMA(diff_1, (na, nb)).fit(trend='nc', disp=0)

    covar = model.cov_params()

    print('ARIMA Estimate 1')
    for i in range(na):
        cov = covar[i][i]
        cov_range = 1.96 * cov ** (1 / 2)
        coef_low = model.params[i] - cov_range
        coef_high = model.params[i] + cov_range
        print(
            f"AR coeficient of order {i + 1} is estimated to be {model.params[i]} with a confidence interval of {coef_low} to {coef_high}")
    for i in range(nb):
        cov = covar[na + i][na + i]
        cov_range = 1.96 * cov ** (1 / 2)
        coef_low = model.params[i] - cov_range
        coef_high = model.params[i] + cov_range
        print(
            f"MA coeficient of order {i + 1} is estimated to be {model.params[na + i]} with a confidence interval of {coef_low} to {coef_high}")

    in_samp_pred = []
    for i in range(len(diff_1)):
        if i == 0:
            temp = diff_1[0]
        elif i == 1:
            er_term = diff_1[0] - in_samp_pred[0]
            temp = model.params[0]*(-1)*diff_1[1] + model.params[2]*er_term
        elif i == 2:
            er_term = model.params[2]*(diff_1[1] - in_samp_pred[1]) + model.params[3]*(diff_1[0] - in_samp_pred[0])
            temp = model.params[0]*(-1)*diff_1[2] + model.params[1]*(-1)*diff_1[1]+ er_term
        else:
            er_term = model.params[2] * (diff_1[i-1] - in_samp_pred[i-1]) + model.params[3] * (diff_1[i-2] - in_samp_pred[i-2])
            temp = model.params[0] * (-1) * diff_1[i-1] + model.params[1] * (-1) * diff_1[i-2] + er_term
        in_samp_pred.append(temp)

    pred = []
    pred_undif = []
    pred_untrans = []
    for i in range(len(series_test)):
        ### Calc Value ###
        if i == 0:
            er_term = model.params[2] * (diff_1[-1] - in_samp_pred[-1]) + model.params[3] * (diff_1[-2] - in_samp_pred[-2])
            temp = model.params[0] * (-1) * diff_1[-1] + model.params[1] * (-1) * diff_1[-2] + er_term
        elif i == 1:
            er_term = model.params[3] * (diff_1[-1] - in_samp_pred[-1])
            temp = model.params[0] * (-1) * pred[0] + model.params[1] * (-1) * diff_1[-1] + er_term
        else:
            er_term = 0
            temp = model.params[0] * (-1) * pred[i-1] + model.params[1] * (-1) * pred[i-2] + er_term
        pred.append(temp)
        ### Untransform ###
        if i == 0:
            df_np = df_log2.to_numpy()
            undif = df_np[-1]+ temp
        else:
            undif = pred_undif[i-1]+temp

        pred_undif.append(undif)
        unlog = math.e**(math.e**(undif))
        pred_untrans.append(unlog)

    diff1_untrans = pd.DataFrame(pred).set_index(series_test.index)
    diff1_pred = pd.DataFrame(pred_untrans).set_index(series_test.index)

    poles_and_roots(model.params, 2, 2)

    na = 4
    nb = 3

    model = sm.tsa.ARMA(diff_1, (na, nb)).fit(trend='nc', disp=0)

    covar = model.cov_params()

    print('ARIMA Estimate 2')
    for i in range(na):
        cov = covar[i][i]
        cov_range = 1.96 * cov ** (1 / 2)
        coef_low = model.params[i] - cov_range
        coef_high = model.params[i] + cov_range
        print(
            f"AR coeficient of order {i + 1} is estimated to be {model.params[i]} with a confidence interval of {coef_low} to {coef_high}")
    for i in range(nb):
        cov = covar[na + i][na + i]
        cov_range = 1.96 * cov ** (1 / 2)
        coef_low = model.params[i] - cov_range
        coef_high = model.params[i] + cov_range
        print(
            f"MA coeficient of order {i + 1} is estimated to be {model.params[na + i]} with a confidence interval of {coef_low} to {coef_high}")

    in_samp_pred = []
    for i in range(len(diff_1)):
        if i == 0:
            temp = diff_1[0]
        elif i == 1:
            er_term = diff_1[0] - in_samp_pred[0]
            temp = model.params[0]*(-1)*diff_1[1] + model.params[4]*er_term
        elif i == 2:
            er_term = model.params[4]*(diff_1[1] - in_samp_pred[1]) + model.params[5]*(diff_1[0] - in_samp_pred[0])
            temp = model.params[0]*(-1)*diff_1[2] + model.params[1]*(-1)*diff_1[1]+ er_term
        elif i == 3:
            er_term = model.params[4]*(diff_1[2] - in_samp_pred[2]) + model.params[5]*(diff_1[1] - in_samp_pred[1]) + model.params[6]*(diff_1[0] - in_samp_pred[0])
            temp = model.params[0]*(-1)*diff_1[2] + model.params[1]*(-1)*diff_1[1] + model.params[2]*(-1)*diff_1[0]+ er_term
        elif i == 4:
            er_term = model.params[4]*(diff_1[3] - in_samp_pred[3]) + model.params[5]*(diff_1[2] - in_samp_pred[2]) + model.params[6]*(diff_1[1] - in_samp_pred[1])
            temp = model.params[0]*(-1)*diff_1[3] + model.params[1]*(-1)*diff_1[2] + model.params[2]*(-1)*diff_1[1]  + model.params[3]*(-1)*diff_1[0] + er_term
        else:
            er_term = model.params[4]*(diff_1[i-1] - in_samp_pred[i-1]) + model.params[5]*(diff_1[i-2] - in_samp_pred[i-2]) + model.params[6]*(diff_1[i-3] - in_samp_pred[i-3])
            temp = model.params[0]*(-1)*diff_1[i-1] + model.params[1]*(-1)*diff_1[i-2] + model.params[2]*(-1)*diff_1[i-3]  + model.params[3]*(-1)*diff_1[i-4] + er_term
        in_samp_pred.append(temp)

    pred = []
    pred_undif = []
    pred_untrans = []
    for i in range(len(series_test)):
        ### Calc Value ###
        if i == 0:
            er_term = model.params[2] * (diff_1[-1] - in_samp_pred[-1]) + model.params[3] * (diff_1[-2] - in_samp_pred[-2])
            temp = model.params[0] * (-1) * diff_1[-1] + model.params[1] * (-1) * diff_1[-2] + er_term
        elif i == 1:
            er_term = model.params[3] * (diff_1[-1] - in_samp_pred[-1])
            temp = model.params[0] * (-1) * pred[0] + model.params[1] * (-1) * diff_1[-1] + er_term
        else:
            er_term = 0
            temp = model.params[0] * (-1) * pred[i-1] + model.params[1] * (-1) * pred[i-2] + er_term
        pred.append(temp)
        ### Untransform ###
        if i == 0:
            df_np = df_log2.to_numpy()
            undif = df_np[-1]+ temp
        else:
            undif = pred_undif[i-1]+temp

        pred_undif.append(undif)
        unlog = math.e**(math.e**(undif))
        pred_untrans.append(unlog)

    diff2_untrans = pd.DataFrame(pred).set_index(series_test.index)
    diff2_pred = pd.DataFrame(pred_untrans).set_index(series_test.index)

    poles_and_roots(model.params, 4, 3)

    plt.figure()
    plt.plot(diff_1, label='Training Set')
    plt.plot(diff1_untrans, label='First (Order 2, 2) ARIMA Model')
    plt.plot(diff2_untrans, label='Second (Order 4, 3) ARIMA Model')
    plt.title('ARIMA Models - Untrans')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(series_train, label='Training Set')
    plt.plot(series_test, label='Test Set')
    plt.plot(diff1_pred, label='First (Order 2, 2) ARIMA Model')
    #plt.plot(diff2_pred, label='Second (Order 4, 3) ARIMA Model')
    plt.title('ARIMA Models - trans')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(series_train, label='Training Set')
    plt.plot(series_test, label='Test Set')
    #plt.plot(diff1_pred, label='First (Order 2, 2) ARIMA Model')
    plt.plot(diff2_pred, label='Second (Order 4, 3) ARIMA Model')
    plt.title('ARIMA Models - trans')
    plt.legend()
    plt.show()

### Part 6 - Regression ###
if run_regression == 1:
    y_train = diff_1
    x_train = frame_train[['DCOILWTICO', 'DGS10', 'DTWEXM', 'Spread']]
    x_train.drop(df.head(1).index,inplace=True)

    x_train_val = x_train.values
    h = np.matmul(x_train_val.T, x_train_val)

    s, d, v = np.linalg.svd(h)
    print("SingleValues = ", d)  # 2 or 3 vars will need to be dropped (of 4?)
    print("The condition number is =", linalg.cond(x_train))  # Very High so a lot of colinearity

    x_train = sm.add_constant(x_train)
    model = sm.OLS(y_train, x_train).fit()
    print(model.summary())

    x_train.drop(['Spread'], axis=1, inplace=True)
    model = sm.OLS(y_train, x_train).fit()
    print(model.summary())

    x_train.drop(['DGS10'], axis=1, inplace=True)
    model = sm.OLS(y_train, x_train).fit()
    print(model.summary())

    x_train.drop(['DTWEXM'], axis=1, inplace=True)
    model = sm.OLS(y_train, x_train).fit()
    print(model.summary())

    x_test = frame_test[['DCOILWTICO', 'DGS10', 'DTWEXM', 'Spread']]
    x_test.drop(['DGS10'], axis=1, inplace=True)
    x_test.drop(['Spread'], axis=1, inplace=True)
    x_test.drop(['DTWEXM'], axis=1, inplace=True)
    x_test = sm.add_constant(x_test)
    reg_pred = model.predict(x_test)
    reg_pred_np = reg_pred.to_numpy()

    pred_undif = []
    pred_untrans = []
    for i in range(len(reg_pred)):
        if i == 0:
            df_np = df_log2.to_numpy()
            undif = df_np[-1]+ reg_pred_np[1]
        else:
            undif = pred_undif[i-1]+reg_pred_np[i]

        pred_undif.append(undif)
        unlog = math.e**(math.e**(undif))
        pred_untrans.append(unlog)

    reg_trans = pd.DataFrame(pred_undif).set_index(series_test.index)
    reg_pred = pd.DataFrame(pred_untrans).set_index(series_test.index)

    plt.figure()
    plt.plot(diff_1, label='Training Set')
    plt.plot(reg_trans, label='Linear Regression Model')
    plt.title('Regression Model - Untransformed')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(series_train, label='Training Set')
    plt.plot(series_test, label='Test Set')
    plt.plot(reg_pred, label='Linear Regression Model')
    plt.title('Regression Model - Transformed')
    plt.legend()
    plt.show()

### Part 7 - Model Compare ###
def autocor(set, lags):

    mean = np.mean(set)
    dif = [obs - mean for obs in set]
    sqr = [(obs-mean)**2 for obs in set]
    sum_sqr = np.sum(sqr)

    lag_set = []
    for x in range(lags+1):
        sum = 0
        for y in range(len(set) - x):
          sum+=dif[y]*dif[y+x]
        acf = sum/sum_sqr
        lag_set.append(acf)

    lag_reverse = lag_set[::-1]
    lag_all = lag_reverse[:-1]+ lag_set
    xx = range(-lags, lags + 1, 1)
    return xx, lag_all

def model_analyze(name, model_pred, num_var):
    if len(model_pred) != len(series_test):
        print("Test length mismatch")
        return -1

    series_test_np = series_test.to_numpy()

    errors = series_test_np - model_pred

    plt.figure()
    xx, lag = autocor(errors, 20)
    plt.stem(xx, lag)
    plt.title(f"{name} Error ACF")
    m = 1.96 / math.sqrt(len(errors))
    plt.axhspan(-m, m, alpha=.1, color='black')
    plt.show()

    mean_err = np.mean(errors)
    err_var = np.var(errors)
    mse = np.mean(errors ** 2)

    print(f"{name}: Mean error: {mean_err}, Error variance: {err_var}, and MSE: {mse}")

    temp = sm.stats.acorr_ljungbox(errors, lags=[20], return_df=True)
    ord1_q = temp['lb_pvalue']
    q_val = ord1_q[20]
    DOF = 20 - num_var
    alfa = 0.01
    chi_critical = chi2.ppf(1 - alfa, DOF)
    if q_val < chi_critical:
        print("Error is white")
    else:
        print("Error is nonwhite")

    means = []
    variances = []

    for i in range(len(errors)):
        temp = errors[:i]
        means.append(np.mean(temp))
        if i > 1:
            variances.append(statistics.variance(temp))
        else:
            variances.append(0)

    plt.figure()
    plt.subplot(211)
    plt.plot(means)
    plt.title(f'Rolling Mean - {name}')
    plt.subplot(212)
    plt.plot(variances)
    plt.title(f'Rolling Variance - {name}')
    plt.show()

    return 0



if run_base == 1 and run_arima == 1 and run_regression == 1:

    avg_pred_np = avg_pred.to_numpy().flatten()
    model_analyze('Base - AVG', avg_pred_np, 2)
    naive_pred_np = naive_pred.to_numpy().flatten()
    model_analyze('Base - Naive', naive_pred_np, 2)
    drift_pred_np = drift_pred.to_numpy().flatten()
    model_analyze('Base - Drift', drift_pred_np, 2)
    ses_pred_np = ses_pred.to_numpy().flatten()
    model_analyze('Base - SES', ses_pred_np, 2)

    holtz_linear_pred_np = holtz_linear_pred.to_numpy().flatten()
    model_analyze('Holts-Linear', holtz_linear_pred_np, 2)
    holtz_winter_pred_np = holtz_winter_pred.to_numpy().flatten()
    model_analyze('Holts-Winter', holtz_winter_pred_np, 2)

    lm1_pred_np = lm1_pred.to_numpy().flatten()
    model_analyze('ARMA Order (1,0)', lm1_pred_np, 1)
    lm2_pred_np = lm2_pred.to_numpy().flatten()
    model_analyze('ARMA Order (2,2)', lm2_pred_np, 4)

    diff1_pred_np = diff1_pred.to_numpy().flatten()
    model_analyze('Differenced ARMA Order (2, 2)', diff1_pred_np, 4)
    diff2_pred_np = diff2_pred.to_numpy().flatten()
    model_analyze('Differenced ARMA Order (4, 3)', diff2_pred_np, 7)

    reg_pred_np = reg_pred.to_numpy().flatten()
    model_analyze('Regression Model', reg_pred_np, 2)