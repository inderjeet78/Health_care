#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 12:42:23 2018

@author: inderjeetsingh
"""

import scipy.integrate as spi
import numpy as np
import pylab as pl
import pandas as pd
import matplotlib.pyplot as plt
death=7.3/(1000*365.0)
birth=19.3/(1000*365.0)
beta=1/10
gamma=1/1000
S0=0.9
I0=0.1

INPUT = (S0, I0)

def diff_eqs(INPUT,t):
    
    Y=np.zeros((2))
    V = INPUT
    Y[0] = - beta * V[0] * V[1] + gamma * V[1] + birth*(V[0]+V[1])-death*V[0]
    Y[1] = beta * V[0] * V[1] - gamma * V[1]-2*death*V[1]
    return Y   # For odeint
time= np.arange(0,366,1)
RES = spi.odeint(diff_eqs,INPUT,time)
#print(RES)
#Ploting
pl.plot(RES[:,0], '-bs', label='Susceptibles')
pl.plot(RES[:,1], '-ro', label='Infectious')
pl.legend(loc=0)
pl.title('SIS epidemic with births and deaths, beta = 1/10 = 100 x gamma')
pl.xlabel('Time')
pl.ylabel('Susceptibles and Infectious')
pl.savefig('sis1.png', dpi=900) # This does increase the resolution.
pl.show()
pd.DataFrame({'Days': time, 'Susceptible': RES[:,0], 'Infected': RES[:,1]}).to_csv("sis1.csv", index=False)




""" Model 2"""


gamma = 1/20
RES = spi.odeint(diff_eqs,INPUT,time)
#print(RES)
#Ploting
pl.plot(RES[:,0], '-bs', label='Susceptibles')
pl.plot(RES[:,1], '-ro', label='Infectious')
pl.legend(loc=0)
pl.title('SIS epidemic with births and deaths, beta = 1/10 = 2 x gamma (Threshold)')
pl.xlabel('Time')
pl.ylabel('Susceptibles and Infectious')
pl.savefig('sis2.png', dpi=900) # This does increase the resolution.
pl.show()
pd.DataFrame({'Days': time, 'Susceptible': RES[:,0], 'Infected': RES[:,1]}).to_csv("sis2.csv", index=False)





""" Model 3 """


gamma = 1/10
RES = spi.odeint(diff_eqs,INPUT,time)
#print(RES)
#Ploting
pl.plot(RES[:,0], '-bs', label='Susceptibles')
pl.plot(RES[:,1], '-ro', label='Infectious')
pl.legend(loc=0)
pl.title('SIS epidemic with births and deaths, beta = 1/10 = gamma')
pl.xlabel('Time')
pl.ylabel('Susceptibles and Infectious')
pl.savefig('sis3.png', dpi=900) # This does increase the resolution.
pl.show()
pd.DataFrame({'Days': time, 'Susceptible': RES[:,0], 'Infected': RES[:,1]}).to_csv("sis3.csv", index=False)

