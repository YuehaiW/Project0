import math
import pymysql
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import coint
from scipy.linalg import toeplitz


def get_data(code):
    conn = pymysql.connect(
        host="192.168.7.93",
        user="quantchina",
        password="zMxq7VNYJljTFIQ8",
        database="wind",
        charset="gbk"
    )
    cursor = conn.cursor()
    df_dict = {}

    for i in range(len(code)):
        query = """
                SELECT TRADE_DT, S_DQ_CLOSE 
                FROM AINDEXEODPRICES 
                WHERE S_INFO_WINDCODE='{ticker}' and TRADE_DT BETWEEN '20140101' AND '20231231' 
                ORDER BY TRADE_DT
            """.format(ticker=code[i])
        cursor.execute(query)

        data = cursor.fetchall()
        df = pd.DataFrame(data, columns=['date', 'prices'])
        df.set_index('date', inplace=True)
        df_dict[code[i]] = df.astype(float)

    df = pd.concat([df_dict[code[0]], df_dict[code[1]]], axis=1)
    df.columns = code
    df['ratio'] = (df[code[1]] / df[code[0]]).astype(float)
    df['mean'] = df.ratio.rolling(window=242).mean()
    df['rate'] = df.ratio.pct_change()
    cursor.close()
    conn.close()

    return df


# HP滤波法去除趋势项
def extract_cycle(y, lamb=1600):
    def diff2(n):
        col = np.zeros(n - 2)
        col[0] = 1
        row = np.zeros(n)
        row[:3] = [1, -2, 1]
        return toeplitz(col, row)

    T = len(y)
    D = diff2(T)
    trend = np.linalg.inv(np.eye(T) + lamb * D.T @ D) @ y
    cycle = y - trend
    return cycle


def At(t: int, lamb):
    e_t = np.zeros(t)
    e_t[-1] = 1

    I_t = np.identity(t - 2)

    # Q_t:二阶差分矩阵，(t-2)xt
    Q_t = np.zeros((t - 2, t))
    for i in range(t - 2):
        Q_t[i, i], Q_t[i, i + 1], Q_t[i, i + 2] = 1, -2, 1

    A_t = np.matrix(e_t) @ Q_t.T @ (np.linalg.inv(Q_t @ Q_t.T + I_t / lamb)) @ Q_t
    return A_t


# 单边HP滤波法，输入的df需要是只有一列的Dataframe
def one_sided_hp_filter(df, lamb):
    df_local = df.copy()
    data_series = np.array(df_local)
    length = len(df)

    list_cycle = [math.nan, math.nan]
    for i in range(2, length):
        # t=i+1
        sub_series = data_series[:i + 1]
        sub_A_t = At(i + 1, lamb)
        cycle_t = (sub_A_t @ sub_series)[0, 0]
        list_cycle.append(cycle_t)
    df_local['cycle_1sHP'] = list_cycle
    df_local['trend_1sHP'] = df[df.columns[0]] - np.array(list_cycle)
    return df_local


def plt_trend(trend, data, lamb):
    plt.plot(data, label='original')
    plt.plot(trend, label='hp trend')
    plt.title('Trend with $\lambda$=' + str(lamb))
    plt.gca().xaxis.set_major_locator(plt.matplotlib.dates.YearLocator(base=1))
    plt.legend()
    plt.show()


# Apply HP filter to extract cycle
df = get_data(['000300.SH', '399852.SZ'])
y1, y2 = df.loc['20171201':, '000300.SH'], df.loc['20171201':, '399852.SZ']
cycle1 = extract_cycle(y1, 129600)
cycle2 = extract_cycle(y2, 129600)

plt_trend(y1 - cycle1, y1, 129600)

# Apply 1s_HP filter to extract cycle
y1_1s = one_sided_hp_filter(pd.DataFrame(y1), 129600)
plt_trend(y1_1s['trend_1sHP'], y1, 129600)

# Cointegration test on cycles
result = coint(cycle1, cycle2)
print("Cointegration Test Results:")
print("Test Statistic:", result[0])
print("P-value:", result[1])
