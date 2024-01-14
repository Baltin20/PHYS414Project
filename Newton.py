import numpy as np
import csv
from matplotlib import pyplot as plt
import scipy.constants as const

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
    
    false_list= mass>threshold
    true_list= mass<=threshold
    not_selected_mass=mass[false_list]
    not_selected_radius=radius[false_list]
    selected_mass=mass[true_list]
    selected_radius=radius[true_list]
    
    return selected_mass,selected_radius,not_selected_mass,not_selected_radius
    
    
    

def newton_b():
    # This is the main function for part b of Newton. This function calculates the R using the 
    # basic formula mg=GMm/r^2. Turns the mass and radius in to the desired scales and then plots
    # the M vs R from the data of 'white_dwarf_data.csv' 
    file_name='white_dwarf_data.csv'
    log_g,mass=read_csv(file_name)
    cgs_g=10**log_g
    mks_g=cgs_g/100
    r=np.sqrt(grav_cons*mass*sun_mass/mks_g)
    r_inearth=r/earth_radi
    plt.scatter(r_inearth,mass)
    plt.ylabel('Mass (in Solar Mass)')
    plt.xlabel('Radius (in Earth Radius)')
    plt.title('M vs R')
    plt.show()
    
    return mass,r_inearth

def newton_c(mass,r):
    
    mass_threshold= 0.404 # in solar mass
    small_mass,small_r,big_mass,big_r=mass_selection(mass, r, mass_threshold)
    # Power-law dependence can be best shown in log-log scale
    log_small_mass,log_small_r,log_big_mass,log_big_r= np.log(small_mass), np.log(small_r), np.log(big_mass), np.log(big_r)
    plt.scatter(log_small_r,log_small_mass,label='Included Data')
    plt.scatter(log_big_r,log_big_mass,label='Excluded Data')

    # Linear Fit would suit the best since relation is in the form logM=logR
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
    plt.ylabel('log(Mass) (in Solar Mass)')
    plt.xlabel('log(Radius) (in Earth Radius)')
    plt.title('M vs R with Line Fit for data with M < '+ str(mass_threshold)+' Solar Mass')
    plt.show()
    print('The slope is ' + str(coefs[0]))
    
    


mass,r=newton_b()
newton_c(mass, r)


  

