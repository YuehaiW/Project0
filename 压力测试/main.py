import numpy as np
import matplotlib.pyplot as plt

from DataGeneration import DataGeneration


def threshold(r, ax):
    """
    做出超额函数与阈值的关系图，帮助选取POT阈值。
    :param r: 收益率
    :param ax: matplotlib Axes 对象
    """
    u_range = np.array(sorted(r[r > 0]))
    e_n = np.zeros(u_range.size)
    for k in range(len(u_range)):
        excess = r[r > u_range[k]] - u_range[k]
        e_n[k] = np.mean(excess)
    ax.scatter(u_range, e_n, s=10)
    ax.set_title("e(u)-u")


ticker_symbols = ['000300.SH', '399852.SZ']
pairs = DataGeneration(ticker_symbols, u_n=-0.026, u_p=0.023, n_sims=1000, method="GPD")
df = pairs.data

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
threshold(df, ax1)
threshold(-df, ax2)
plt.show()

pairs.monte_carlo_sim()
