import numpy as np
import pymysql
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import norm, genpareto


class DataGeneration:
    """
    根据所给数据使用Peak Over Threshold(POT)方法生成一系列日收益率数据
    """

    def __init__(self, tickers, u_n, u_p, n_sims=100, method='GPD'):
        """
        tickers: 用于拟合分布参数的股票代码
        u_n: 负向阈值，当损失超过该阈值时超额损失服从广义帕累托分布(GPD)
        u_p: 正向阈值，当收益超过该阈值时超额收益服从广义帕累托分布(GPD)
        n_sims: 蒙特卡罗模拟的次数
        method: 生成数据所使用的方法
        """

        self.tickers = tickers
        self.u_n, self.u_p = u_n, u_p
        self.n_sims = n_sims
        self.method = method

        self.data = self.get_data()
        self.lamb_p, self.lamb_n = self.poisson_est()
        self.norm_mu, self.norm_std = self.normal_est()
        self.gpd_params = self.genpareto_est()

        self.actual_values = pd.concat([pd.Series([1]), (1 + self.data).cumprod()],
                                       axis=0, ignore_index=True).to_numpy()
        self.sim_values = np.zeros((self.n_sims, len(self.data) + 1))

    def get_data(self):
        """
        获取数据，输入一只股票时按其收益率建模；输入两只股票时按配对交易思想去比值收益率建模。
        """
        code = self.tickers
        conn = pymysql.connect(
            host="192.168.7.93",
            user="quantchina",
            password="zMxq7VNYJljTFIQ8",
            database="wind",
            charset="gbk"
        )
        cursor = conn.cursor()

        if len(code) == 1:
            query = """
                SELECT TRADE_DT, S_DQ_PCTCHANGE 
                FROM AINDEXEODPRICES 
                WHERE S_INFO_WINDCODE='{ticker}' and TRADE_DT BETWEEN '20190101' AND '20231231' 
                ORDER BY TRADE_DT
            """.format(ticker=code[0])
            cursor.execute(query)
            data = cursor.fetchall()

            df = pd.DataFrame(data, columns=['date', 'returns'])
            df.set_index('date', inplace=True)
            df['returns'] = df['returns'].astype(float) / 100
            cursor.close()
            conn.close()

            return df.returns
        elif len(code) == 2:
            df_dict = {}
            for i in range(len(code)):
                query = """
                    SELECT TRADE_DT, S_DQ_CLOSE 
                    FROM AINDEXEODPRICES 
                    WHERE S_INFO_WINDCODE='{ticker}' and TRADE_DT BETWEEN '20190101' AND '20231231' 
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
            df['rate'] = df.ratio.pct_change()
            cursor.close()
            conn.close()

            return df.rate.dropna()
        else:
            cursor.close()
            conn.close()
            print("Unsupported data. Please input less than 2 tickers.")

    def poisson_est(self):
        """估计正向和负向极值发生的概率。"""
        ser = self.data
        total_nums = len(ser)
        positive_nums = len(ser[ser > self.u_p])
        negative_nums = len(ser[ser < self.u_n])
        lambda_p, lambda_n = positive_nums / total_nums, negative_nums / total_nums
        return lambda_p, lambda_n

    def normal_est(self):
        """估计极值不发生时正态分布的参数。"""
        ser = self.data
        ser_mid = ser[(ser >= self.u_n) & (ser <= self.u_p)]
        norm_mu, norm_std = norm.fit(ser_mid.values)
        return norm_mu, norm_std

    def genpareto_est(self):
        """分别估计正向和负向极值的广义Pareto分布参数。"""
        ser = self.data
        df_p, df_n = ser[ser > self.u_p], ser[ser < self.u_n]
        params_p = genpareto.fit(df_p.values, floc=0)
        params_n = genpareto.fit(-df_n.values, floc=0)
        params_df = pd.DataFrame({'shape': [params_p[0], params_n[0]], 'scale': [params_p[2], params_n[2]]},
                                 index=['positive', 'negative'])
        return params_df

    def data_generate(self):
        """根据正态分布或GPD分布模拟生成一段时间的日收益率。"""
        res = []
        if self.method == 'GPD':
            p = [1 - self.lamb_p - self.lamb_n, self.lamb_p, self.lamb_n]
            for i in range(len(self.data)):
                rand_num = np.random.choice([0, 1, 2], p=p)
                if rand_num == 0:
                    sample = np.random.normal(self.norm_mu, self.norm_std, 1)
                elif rand_num == 1:
                    sample = genpareto.rvs(c=self.gpd_params.iloc[0, 0], loc=0,
                                           scale=self.gpd_params.iloc[0, 1], size=1) + self.u_p
                else:
                    sample = -genpareto.rvs(c=self.gpd_params.iloc[1, 0], loc=0,
                                            scale=self.gpd_params.iloc[1, 1], size=1) + self.u_n
                res.append(sample)
        elif self.method == 'norm':
            sample = np.random.normal(self.norm_mu, self.norm_std, len(self.data))
            res = sample.tolist()
        else:
            print("Unsupported method. Please choose GPD or norm!")
            return

        return res

    def monte_carlo_sim(self):
        """依照设定的模拟次数生成一系列收益数据。"""
        for i in range(self.n_sims):
            sim_returns = [0] + self.data_generate()
            scenario_returns = pd.DataFrame(sim_returns)
            scenario_prices = (1 + scenario_returns).cumprod()
            self.sim_values[i, :] = scenario_prices.to_numpy().flatten()

        for i in range(self.n_sims):
            plt.plot(self.sim_values[i, :], linewidth=0.5, alpha=0.3)

        if self.method == 'GPD':
            plt.title('Monte Carlo Simulation - EVT Stress Testing')
        else:
            plt.title('Monte Carlo Simulation - Normal cases')

        plt.xlabel('Time')
        plt.ylabel('Scenario Accumulative Return Rate')
        plt.grid(True)

        plt.plot(self.actual_values, label='Actual Return')
        plt.legend(loc='upper left')
        plt.show()

        # return np.concatenate((self.actual_values, self.sim_values))


if __name__ == '__main__':
    def get_rank(actual, sim):
        arr = [actual] + sim.tolist()
        sorted_arr = sorted(arr)
        rank = sorted_arr.index(actual) + 1
        return rank


    def deviations(actual, sim):
        mean = np.mean(sim)
        std = np.std(sim)
        return (actual - mean) / std


    # Example of usage #

    ticker_symbols = ['000300.SH', '399852.SZ']
    obj_gpd = DataGeneration(ticker_symbols, u_n=-0.026, u_p=0.023, n_sims=1000, method="GPD")
    obj_norm = DataGeneration(ticker_symbols, u_n=-0.026, u_p=0.023, n_sims=1000, method="norm")
    obj_gpd.monte_carlo_sim()
    obj_norm.monte_carlo_sim()

    r_gpd = get_rank(obj_gpd.actual_values[-1], obj_gpd.sim_values[:, -1])
    d_gpd = deviations(obj_gpd.actual_values[-1], obj_gpd.sim_values[:, -1])

    r_norm = get_rank(obj_norm.actual_values[-1], obj_norm.sim_values[:, -1])
    d_norm = deviations(obj_norm.actual_values[-1], obj_norm.sim_values[:, -1])

    print("mean return rate: ", np.mean(obj_gpd.sim_values[:, -1]))
    print("std return rate: ", np.std(obj_gpd.sim_values[:, -1]))
    print("GPD rank: ", r_gpd)
    print("GPD deviation: ", d_gpd)
    print("mean return rate: ", np.mean(obj_norm.sim_values[:, -1]))
    print("std return rate: ", np.std(obj_norm.sim_values[:, -1]))
    print("Norm rank: ", r_norm)
    print("Norm deviation: ", d_norm)

    sns.histplot(obj_gpd.sim_values[:, -1] - 1, label="gpd")
    sns.histplot(obj_norm.sim_values[:, -1] - 1, label="norm")
    plt.axvline(x=obj_gpd.actual_values[-1] - 1, color='red', label="actual")
    plt.legend()
    plt.show()


