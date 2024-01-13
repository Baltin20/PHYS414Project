import numpy as np
import csv
from matplotlib import pyplot as plt
import scipy.constants as const

grav_cons=const.G
sun_mass=(1.989)*10**30 # in kg
earth_radi=(6371)*10**3 # in m


def read_csv(file_name):
    # Takes the comma seperated file's name as an input and extracts the surface gravity in CGS base 10 and mass in solar mass 
    # and return these values as a list
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

newton_b()
  

