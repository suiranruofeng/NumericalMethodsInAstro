import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# 定义一些物理常量
M_SUN = 1.9891e33  # 太阳质量, 单位: 克
Mpc = 3.08567758e24  # 兆parsec, 单位: 厘米
m_H = 1.6735575e-24  # 氢原子质量, 单位: 克
yr = 31536000  # 一年的秒数

# 定义宇宙学参数
Omega_m = 0.32  # 物质密度参数
Omega_Lambda = 1 - Omega_m  # 暗能量密度参数
h = 0.67  # Hubble参数的无量纲系数
H0 = 100 * h  # Hubble常数, 单位: km/s/Mpc

# 定义再电离相关参数
f_esc = 0.5  # 电离光子逃逸概率
n_H = 1.9e-7  # 氢原子数密度, 单位: 厘米^-3
clumping_factor = 3.0  # 团块因子
alpha_B = 2.5e-13  # 三体重复复合系数, 单位: 厘米^3/秒
N_ion = 4000.0  # 每个星系形成的电离光子数

# 定义星系形成率密度函数
def SFRD(z):
    a = 0.015
    b = 2.7
    c = 2.9
    d = 5.6
    return a * (1 + z) ** b / (1 + ((1 + z) / c) ** d)

# 定义 Hubble 参数函数
def Hz(z):
    return H0 * np.sqrt(Omega_m * (1 + z) ** 3 + Omega_Lambda)

# 定义红移变化率函数
def dz_dt(z):
    return -(1 + z) * Hz(z) * 1e5 / Mpc

# 定义电离泡充满因子 Q_I 对红移的变化率函数
def dQI_dz(z, QI):
    dot_n_ion = N_ion * SFRD(z) * M_SUN / yr / (Mpc ** 3) / m_H  # 电离光子产生率, 单位: 厘米^-3/秒
    dQI_dt = f_esc * dot_n_ion / n_H - clumping_factor * alpha_B * n_H * (1 + z) ** 3 * QI  # 电离泡充满因子变化率, 单位: 每秒
    return dQI_dt / dz_dt(z)

# 绘制电离泡充满因子随红移的变化
plt.figure()
t_eval = np.linspace(30.0, 5.0, 201)  # 计算红移范围为 30 到 5
sol = solve_ivp(dQI_dz, [30.0, 5.0], [0.0], method='RK45', t_eval=t_eval)  # 求解微分方程

# 处理解的边界条件
index = np.where(sol.y[0][:] > 1)[0]
sol.y[0][index] = 1

# 绘制结果
plt.plot(sol.t, sol.y[0], linewidth=2, color='r', linestyle='-', label=r'ionized bubble filling factor')   

# 设置坐标轴和图例
ax = plt.gca()
plt.xlim([5.0, 30.0])
plt.ylim([0.0, 1.02])
plt.xlabel(r'$z$', fontsize=20)
plt.ylabel(r'$Q_{\rm I}(z)$', fontsize=20)
plt.setp(ax.get_xticklabels(), fontsize=20)
plt.setp(ax.get_yticklabels(), fontsize=20)
plt.legend(loc='upper right', frameon=False, fontsize=10)

# 保存图像
plt.savefig('./reionization.pdf', bbox_inches='tight')