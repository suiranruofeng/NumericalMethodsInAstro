import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


M_SUN=1.9891e33 # g
Mpc=3.08567758e24 # cm
m_H=1.6735575e-24 # g
yr=31536000 # second

Omega_m=0.32
Omega_Lambda=1-Omega_m
h=0.67
H0=100*h

f_esc=0.5

n_H=1.9e-7 # cm^{-3}

clumping_factor=3.0

alpha_B=2.5e-13 #cm^3/s

N_ion=4000.0

# M_SUN/yr/Mpc^3
def SFRD(z):
    a=0.015
    b=2.7
    c=2.9
    d=5.6
    return a*(1+z)**b/(1+((1+z)/c)**d )


def Hz(z):
    return H0*np.sqrt(Omega_m*(1+z)**3+Omega_Lambda)

# /s
def dz_dt(z):
    return -(1+z)*Hz(z)*1e5/Mpc

def dQI_dz(z,QI):
    dot_n_ion=N_ion*SFRD(z)*M_SUN/yr/(Mpc**3)/m_H # /cm^3/s
    dQI_dt=f_esc*dot_n_ion/n_H-clumping_factor*alpha_B*n_H*(1+z)**3*QI # /s
    return dQI_dt/dz_dt(z)



plt.figure()

t_eval=np.linspace(30.0,5.0,201)

sol=solve_ivp(dQI_dz,[30.0,5.0],[0.0],method='RK45',t_eval=t_eval)

index=np.where(sol.y[0][:]>1)[0]
sol.y[0][index]=1

plt.plot(sol.t,sol.y[0],linewidth=2,color='r',linestyle='-',label=r'ionized bubble filling factor')   
####


ax=plt.gca()

plt.xlim([5.0,30.0])
plt.ylim([0.0,1.02])

plt.xlabel(r'$z$',fontsize=20)
plt.ylabel(r'$Q_{\rm I}(z)$',fontsize=20)


plt.setp(ax.get_xticklabels(),fontsize=20)
plt.setp(ax.get_yticklabels(),fontsize=20)

plt.legend(loc='upper right',frameon=False,fontsize=10)
 
plt.savefig('./reionization.pdf',bbox_inches='tight')

 
