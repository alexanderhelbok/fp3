import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
from uncertainties import ufloat

plt.style.use('Source.mplstyle')

def parabola(x, a_param, h, k):
    return a_param * (x - h) ** 2 + k

df1, df2, df3, df4 = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

a = np.arange(30, 61, 2)
counts = np.array([2282, 2325, 2464, 2684, 2664, 2831, 2929, 2825, 2952, 2863, 2837, 2694, 2629, 2398, 2387, 2135]) - 30
error = np.sqrt(counts)
df1['x'] = a
df1['counts'] = counts
df1['error'] = error

a = np.arange(62, 111, 4)
counts = np.array([2187, 2194, 2415, 2633, 2756, 2689, 2685, 2726, 2589, 2536, 2296, 2159, 1983]) -30
error = np.sqrt(counts)
df2['x'] = a
df2['counts'] = counts
df2['error'] = error

a = np.arange(74, 101, 2)
counts = np.array([2102, 2214, 2310, 2476, 2547, 2734, 2612, 2738, 2708, 2604, 2580, 2480, 2344, 2284]) - 30
error = np.sqrt(counts)
df3['x'] = a
df3['counts'] = counts
df3['error'] = error

a = np.arange(70, 109, 4)
counts = np.array([1993, 2289, 2309, 2438, 2436, 2395, 2456, 2235, 2171, 1983])
error = np.sqrt(counts)
df4['x'] = a
df4['counts'] = counts
df4['error'] = error


fig, ax = plt.subplots(2, 2, figsize=(7.5, 5))
ax = ax.flatten()

for i, df in enumerate([df1, df2, df3, df4]):
    plot_x = np.linspace(min(df['x']), max(df['x']), 1000)

    # print((df['x'].iloc[-1] + df['x'].iloc[0])/2)
    para, covm = curve_fit(parabola, df.x, df.counts, sigma=df.error, absolute_sigma=True, p0=[0, (df['x'].iloc[-1] + df['x'].iloc[0])/2, 2000])
    err = np.sqrt(np.diag(covm))
    print(f"a_param: {ufloat(para[0], err[0]):.1uS}, h: {ufloat(para[1], err[1]):.1uS}, k: {ufloat(para[2], err[2]):.1uS}")

    ax[i].errorbar(df.x, df.counts, yerr=df.error, fmt='.k', label='Data', capsize=3)
    ax[i].plot(plot_x, parabola(plot_x, *para))
    ax[i].set_xlabel('Angle (degrees)')
    ax[i].set_ylabel('Counts')

    # ax[i].legend()

# para, covm = curve_fit(parabola, a, counts, sigma=error, absolute_sigma=True)

# plt.errorbar(a, counts, yerr=error, fmt='o', label='Data', capsize=3)
# plt.plot(plot_x, parabola(plot_x, *para))
# plt.xlabel('a')
# plt.ylabel('Counts')

# plt.legend()

plt.tight_layout()
# plt.savefig('optimize.pdf', bbox_inches='tight') 
plt.show()

