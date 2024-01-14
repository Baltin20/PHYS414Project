import numpy as np
from matplotlib import pyplot as plt
import scipy.constants as const
from scipy.integrate import solve_ivp

grav_cons=const.G
sun_mass=(1.989)*10**30 # in kg
earth_radi=(6371)*10**3 # in m
k_ns=100
length_scale=1477 #in m
time_scale=(4.927)*10**-6 # in s


def tov_RHS(t,y,k_ns):
    # t stands for r here
    RHS=np.zeros(y.shape[0])
    rho=np.sqrt(y[2]/k_ns)
    RHS[0]=4*np.pi*(t**2)*rho
    if t==0:
        RHS[1]=0
        RHS[2]=0
    else:
        RHS[1]=2*(y[0]+4*np.pi*(t**3)*y[2])/(t*(t-2*y[0]))
        RHS[2]=-RHS[1]*(rho+y[2])/2
        
    return RHS.flatten()

def stop_sign(t,y,k_ns):
    return y[2]

stop_sign.terminal= True
stop_sign.direction=-1


def tov_solver(p_c,r_lim=30):
    step_size=10**-3
    mass_init=0
    dilation_init=0
    pressure_init=p_c
    init=np.array([mass_init,dilation_init,pressure_init]).flatten()
    solution=solve_ivp(tov_RHS,[0,r_lim],init,max_step=step_size,events=stop_sign,args=([int(k_ns)]))
    radius=solution.t[-1]
    mass=solution.y[0,-1]
    return mass,radius

