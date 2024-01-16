import numpy as np
import csv
from matplotlib import pyplot as plt
import scipy.constants as const
from scipy.integrate import solve_ivp

grav_cons=const.G
sun_mass=(1.989)*10**30 # in kg
earth_radi=(6371)*10**3 # in m



def read_csv(file_name):
    # Takes the comma seperated file's name as an input and extracts the surface gravity in CGS log base 10 and mass in solar mass 
    # and return these values as a np array
    name=[]
    logg=[] 
    mass=[]
    with open(file_name,'r') as file:
        reader=csv.reader(file)
        next(reader) # in order to skip header
        for row in reader:
            name.append(row[0])
            logg.append(float(row[1]))
            mass.append(float(row[2]))
    return np.asarray(logg),np.asarray(mass)

def mass_selection(mass,radius,threshold):
    # Separates the mass and radius depending on the threshold, returns the data to include and exclude
    
    false_list= mass>threshold
    true_list= mass<=threshold
    not_selected_mass=mass[false_list]
    not_selected_radius=radius[false_list]
    selected_mass=mass[true_list]
    selected_radius=radius[true_list]
    
    return selected_mass,selected_radius,not_selected_mass,not_selected_radius

def lane_emden_RHS(t,y,n):
    # t corresponds to xi. Creates the right hand side array from the previous solution and returns it to be fed to solver. n must be fed through the args
    # y[0]= theta, y[1]= dtheta
    RHS=np.zeros(y.shape[0])
    RHS[0]=y[1]
    if t == 0 :
        RHS[1]=0
    else:
        RHS[1]=-(2/t)*y[1]-y[0]**n
        
    return RHS.flatten()
    # No need for explicit event because when y[0] = theta overshoots 0 the else statement gives warning and the computation stops see the last computed theta_n value is almost 0
    # if i handle the warning coming from negative y[0] then the solver automatically stops
    
def lane_emden_solver(n,xi_lim=15):
    #Real lane_emden solver the xi_lim was arbitrarily given ideally the solver stops before hitting that value
    # This function evolves theta and dtheta in xi given an n value
    
    step_size=10**-3
    theta_0=1
    dtheta_0=0
    init=np.array([theta_0,dtheta_0]).flatten() #solve_ivp requires flattened arrays
    solution=solve_ivp(lane_emden_RHS,[0,xi_lim],init,max_step=step_size,args=([float(n)]))
    xi=solution.t
    theta=solution.y[0,:]
    dtheta=solution.y[1,:]
    
    return xi,theta,dtheta  

def newton_b():
    # This is the main function for part b of Newton. This function calculates the R using the 
    # basic formula mg=GMm/r^2. Turns the mass and radius in to the desired scales and then plots
    # the M vs R from the data of 'white_dwarf_data.csv' 
    file_name='white_dwarf_data.csv'
    log_g,mass=read_csv(file_name)
    cgs_g=10**log_g
    mks_g=cgs_g/100
    r=np.sqrt(grav_cons*mass*sun_mass/mks_g)# rhs is all mks
    r_inearth=r/earth_radi
    
    #Plotting
    plt.scatter(r_inearth,mass)
    plt.ylabel('Mass (in Solar Mass)')
    plt.xlabel('Radius (in Earth Radius)')
    plt.title('M vs R')
    plt.show()
    
    return mass,r_inearth # these parts are required for other newton functions

def newton_c(mass,r):
    # This is the main function for part c of Newton. This function chooses the data to use as small mass WD, plots the log-log function including the chosen data
    # creates a linear fit for selected data, solves Lane-Emden Equation.
    # input mass and r are in Solar mass and earth radius respectively.
    
    mass_threshold= 0.404 # in solar mass
    small_mass,small_r,big_mass,big_r=mass_selection(mass, r, mass_threshold)
    # Power-law dependence can be best shown in ln-ln scale
    log_small_mass,log_small_r,log_big_mass,log_big_r= np.log(small_mass), np.log(small_r), np.log(big_mass), np.log(big_r)
    plt.scatter(log_small_r,log_small_mass,label='Included Data')
    plt.scatter(log_big_r,log_big_mass,label='Excluded Data')

    # Linear Fit would suit the best since relation is in the form lnM=lnR
    coefs=np.polyfit(log_small_r,log_small_mass,1)
    slope,intercept=coefs

    # The span for the fit is being chosen
    x_max=np.min(log_big_r)
    x_min=np.max(log_small_r)
    x_span=np.linspace(x_min, x_max,len(r))
    line_fit=slope*x_span+intercept

    #Plotting
    plt.plot(x_span,line_fit,color='red',linestyle='--',label='Line Fit')  
    plt.legend()
    plt.ylim(min(log_small_mass)-0.1,max(log_big_mass)+0.1)
    plt.ylabel('ln(Mass) (in Solar Mass)')
    plt.xlabel('ln(Radius) (in Earth Radius)')
    plt.title('M vs R with Line Fit for data with M < '+ str(mass_threshold)+' Solar Mass')
    plt.show()
    print('The slope is ' + str(coefs[0]))
    n=1.5
    int_slope=-3
    
    with np.errstate(invalid='ignore'): # as mentioned warning handling 
        xi,theta,dtheta=lane_emden_solver(n,15)
    # Those are the last values 
    xi_n=xi[-1]
    theta_n=theta[-1]
    dtheta_n=dtheta[-1]
    grav_cons_scaled=grav_cons*sun_mass/(earth_radi**3)
    k_val=(4*np.pi*grav_cons_scaled/(2.5))*np.power((np.exp(intercept)/(-4*np.pi*(xi_n**5)*dtheta_n)),1/3)
    # the mks conversion requires + log(sun_mass)-slope*log(earth radi) to the intercept since -> ln(m)+ln(ms)=ln(B)+s*lnr+s*ln(re) -> intercept = ln(B)+s*ln(re)-ln(ms) in solar mass, earth radius
    k_val_mks=(4*np.pi*grav_cons/(2.5))*np.power((np.exp(intercept+np.log(sun_mass)-int_slope*np.log(earth_radi))/(-4*np.pi*(xi_n**5)*dtheta_n)),1/3)
    
    print('xi_n is:' +str(xi_n))
    print('theta_n is: ' +str(theta_n))
    print('dtheta_n is: ' +str(dtheta_n))
    
    
    print('The K* value is ' + str(k_val)+' scaled with respect to solar mass and earth radius' )
    print('The K* value is ' + str(k_val_mks)+' in mks' )
    
    #This can be obtained from the total mass equation
    rho_c= (-small_mass*xi_n)/(4*dtheta_n*np.pi*np.power(small_r,3))
    plt.scatter(small_mass,rho_c)
    plt.ylabel('Central Density (in Solar Mass / Earth Radius^3)')
    plt.xlabel('Mass (in Solar Mass)')
    plt.title('Central Density vs Mass')
    plt.show()
       
def final():
    #main function to execute newton functions

    mass,r=newton_b()
    newton_c(mass, r)

final()



  

