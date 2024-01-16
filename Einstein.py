import numpy as np
#Einstein.py
from matplotlib import pyplot as plt
import scipy.constants as const
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.optimize import newton

grav_cons=const.G
sun_mass=(1.989)*10**30 # in kg
earth_radi=(6371)*10**3 # in m
k_ns=100
length_scale=1477 #in m
time_scale=(4.927)*10**-6 # in s


def dm(mass):
    # Taking a vector returns its first derivative with 3 point stencil
    l_shift=(1/2)*np.roll(mass,-1) # corresponds to +drho
    r_shift=-(1/2)*np.roll(mass,1) # corresponds to -drho
    dm=l_shift+r_shift
    dm[0]=-(3/2)*mass[0]+2*mass[1]-(1/2)*mass[2]
    dm[-1]=(3/2)*mass[-1]-2*mass[-2]+(1/2)*mass[-3]
    
    return dm
        

def tov_RHS(t,y,k_ns):
    # t stands for r. This function returns the RHS of ODES. It also includes the RHS of baryonic mass ODE
    #y[0]=mass,y[1]=time dilation,y[2]=pressure,y[3]=baryonic mass
    RHS=np.zeros(y.shape[0])
    with np.errstate(invalid='ignore'): # p overshoots 0 and stop but last value results in warning. For output to be aesthetic I catch the warning
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
    # This is an event function that must stop the calculation when the sign changes
    # I realized since the inside of square root of rho becomes - (tov_RHS function) the tov_solver automatically stops, I was only required to handle warning but better to be safe
    return y[2]

stop_sign.terminal= True
stop_sign.direction=-1


def tov_solver(p_c,k,r_lim=30):
    # This is the solver that evolves mass,time_dilation_pressure and baryonic mass in r
    # it returns the last value of mass,radius and baronic mass
    # step size was not specified this time
    mass_init=0
    dilation_init=0
    pressure_init=p_c
    baryonic_mass_init=0
    init=np.array([mass_init, dilation_init, pressure_init,baryonic_mass_init]).flatten()
    solution=solve_ivp(tov_RHS,[0,r_lim],init,events=stop_sign,args=([int(k)]))
    radius=solution.t[-1]
    mass=solution.y[0,-1]
    baryonic_mass=solution.y[3,-1]  
    
    return mass,radius,baryonic_mass

def einstein_a():
    # The main function for einstein part a iterating over the pressure finds mass, radius,baryonic_mass
    # Return thhose arrays therefore it is important to be computed first
    
    n_sample=100
    rho_start=10**-5
    rho_stop=10**-2
    # Initialization
    rho_c_span=np.linspace(rho_start,rho_stop,n_sample) 
    pressure_c_span=k_ns*(rho_c_span**2)
    mass=np.zeros_like(pressure_c_span)
    radius=np.zeros_like(pressure_c_span)
    baryonic_mass=np.zeros_like(pressure_c_span)
    
    for i in range(pressure_c_span.shape[0]):
        mass[i],radius[i],baryonic_mass[i]=tov_solver(pressure_c_span[i],k_ns,r_lim=30)

    plt.plot(radius*length_scale/1000,mass)# to use km
    plt.ylabel('Mass (in Solar Mass)')
    plt.xlabel('Radius (in km)')
    plt.title('M vs R')
    plt.show()
    return mass,radius,baryonic_mass

def einstein_b(mass,radius,baryonic_mass):
    # The main function for einstein part b, uses the already computed mass, radius and baryonic masses to plot
   
    frac_bind_energy=(baryonic_mass-mass)/mass
    plt.plot(radius*length_scale/1000,frac_bind_energy)
    plt.ylabel('Fractional Binding Energy')
    plt.xlabel('Radius (in km)')
    plt.title(r'$\Delta$ vs R')
    plt.show()

def einstein_c(mass):
    # The main function for einstein part c,uses already computed mass.
    # Finds the stable and unstable masses depending on the sign of derivative and the max stable mass
    
    n_sample=100
    rho_c_span=np.linspace(10**-5,10**-2,n_sample) 
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
    print('Max NS Mass allowed: '+ str(np.max(stable_mass))+' in Solar Mass.')
    
def einstein_d():
    # The main function for einstein part d, For every value of k iterates over the pressure and finds the maximum stable mass for specific K value
    # then from that values plots a max mass vs K curve
    
    k_min=20
    k_max=180
    n_sample=40 # Computatinally costly operatioons therefore n is small
    k_span=np.linspace(k_min,k_max,n_sample)
    rho_c_span=np.linspace(10**-5,10**-2,n_sample)
    max_mass=np.zeros_like(k_span)
    pressure_c_span=k_ns*(rho_c_span**2)
    for j in range(k_span.shape[0]):
        mass=np.zeros_like(rho_c_span)
        for i in range(pressure_c_span.shape[0]):
            mass[i],a,b=tov_solver(pressure_c_span[i],k_span[j],r_lim=30) #radius and baryon mass not important
        drho=rho_c_span[1]-rho_c_span[0]
        dmass=dm(mass)/drho
        true_list=dmass>0
        stable_mass,stable_rho=mass[true_list],rho_c_span[true_list]
        max_mass[j]=np.max(stable_mass)

    plt.scatter(k_span,max_mass,label='Max Mass') # the sampling size is small therefore instead of plot I will be using cubic spline
    cubic_spline=interp1d(k_span, max_mass,kind='cubic')# furthermore cubic spline is required to use in root finnding to find max K value possible for the observed max mass
    spline_span=np.linspace(k_min,k_max,n_sample*100)    
    plt.plot(k_span,cubic_spline(k_span),color='red',label='Cubic Spline')
    plt.legend()
    plt.xlabel('K values')
    plt.ylabel('Max NS Mass (in Solar Mass)')
    plt.title('Max Mass vs K')
    plt.show()
    observed_m=2.14
    root_to_find= lambda x: cubic_spline(x)-observed_m
    max_k=newton(root_to_find,110)# the value 110 was chosen from the plot
    print('Max k allowed: '+ str(max_k))
    
    

def final():
    #main function to execute newton functions

    mass,radius,baryonic_mass=einstein_a()
    einstein_b(mass, radius, baryonic_mass) 
    einstein_c(mass)
    einstein_d()


    



    
    
    
    
    
    
    
    
    