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


def dm(mass):
    # 
    l_shift=(1/2)*np.roll(mass,-1) # corresponds to +drho
    r_shift=-(1/2)*np.roll(mass,1) # corresponds to -drho
    dm=l_shift+r_shift
    dm[0]=-(3/2)*mass[0]+2*mass[1]-(1/2)*mass[2]
    dm[-1]=(3/2)*mass[-1]-2*mass[-2]+(1/2)*mass[-3]
    
    return dm
        

def tov_RHS(t,y,k_ns):
    # t stands for r here
    RHS=np.zeros(y.shape[0])
    rho=np.sqrt(y[2]/k_ns)
    RHS[0]=4*np.pi*(t**2)*rho
    if t==0:
        RHS[1]=0
        RHS[2]=0
        RHS[3]=0
    else:
        RHS[1]=2*(y[0]+4*np.pi*(t**3)*y[2])/(t*(t-2*y[0]))
        RHS[2]=-RHS[1]*(rho+y[2])/2
        RHS[3]=4*np.pi*np.power(1-(2*y[0]/t),-1/2)*(t**2)*rho
        
    return RHS.flatten()

def stop_sign(t,y,k_ns):
    return y[2]

stop_sign.terminal= True
stop_sign.direction=-1


def tov_solver(p_c,r_lim=30):

    mass_init=0
    dilation_init=0
    pressure_init=p_c
    baryonic_mass_init=0
    init=np.array([mass_init, dilation_init, pressure_init,baryonic_mass_init]).flatten()
    solution=solve_ivp(tov_RHS,[0,r_lim],init,events=stop_sign,args=([int(k_ns)]))
    radius=solution.t[-1]
    mass=solution.y[0,-1]
    baryonic_mass=solution.y[3,-1]  
    
    return mass,radius,baryonic_mass

def einstein_abc():
    n_sample=100
    rho_c_span=np.linspace(10**-5,10**-2,n_sample) 
    pressure_c_span=k_ns*(rho_c_span**2)
    mass=np.zeros_like(pressure_c_span)
    radius=np.zeros_like(pressure_c_span)
    baryonic_mass=np.zeros_like(pressure_c_span)
    for i in range(pressure_c_span.shape[0]):
        mass[i],radius[i],baryonic_mass[i]=tov_solver(pressure_c_span[i],r_lim=30)

    plt.plot(radius*length_scale/1000,mass)
    plt.ylabel('Mass (in Solar Mass)')
    plt.xlabel('Radius (in km)')
    plt.title('M vs R')
    plt.show()

    frac_bind_energy=(baryonic_mass-mass)/mass
    plt.plot(radius*length_scale/1000,frac_bind_energy)
    plt.ylabel('Fractional Binding Energy')
    plt.xlabel('Radius (in km)')
    plt.title(r'$\Delta$ vs R')
    plt.show()

    """
    plt.scatter(rho_c_span,mass) # I can clearly see that there is a maximum point and after that the solution is unstable
                                 #But I assume instructor want me to use a numerical differentiation otherwise it would be easy task therefore I won't be abusing the graph
    plt.show()
    """
    drho=rho_c_span[1]-rho_c_span[0]
    dmass=dm(mass)/drho
    true_list=dmass>0
    false_list=dmass<0
    stable_mass,stable_rho=mass[true_list],rho_c_span[true_list]
    unstable_mass,unstable_rho=mass[false_list],rho_c_span[false_list]
    plt.plot(stable_rho*sun_mass/(length_scale**3),stable_mass,label='Stable Mass')
    plt.plot(unstable_rho*sun_mass/(length_scale**3),unstable_mass,label='Unstable Mass')
    plt.ylabel('Mass (in Solar Mass)')
    plt.xlabel('Central Density (in kg/$m^3$)')
    plt.title(r'M vs $\rho_c$')
    
    plt.legend()
    plt.show()
    
einstein_abc() 
    



    
    
    
    
    
    
    
    
    