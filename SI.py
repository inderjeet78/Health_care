#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 18:38:05 2018

@author: inderjeetsingh
"""

import scipy.integrate as spi
import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
import pandas as pd

death=7.3/(1000*365.0)
birth=19.3/(1000*365.0)
beta=1/10
gamma=1/1000


TS = 1
ND=1*365
S0=0.5
I0=0.4
R0=1-S0-I0
INPUT = (S0, I0)

def diff_eqs(INP,t):
    
    Y=np.zeros((2))
    V = INP
    Y[0] = - beta * V[0] * V[1] + birth-death*V[0]
    Y[1] = beta * V[0] * V[1] - gamma * V[1]-death*V[1]
    return Y   # For odeint
t_start = 0.0; t_end = ND; t_inc = TS
t_range = np.arange(t_start, t_end+t_inc, t_inc)
RES = spi.odeint(diff_eqs,INPUT,t_range)
#print(RES)
#Ploting
pl.plot(RES[:,0], '-bs', label='Susceptibles')
pl.plot(RES[:,1], '-ro', label='Infectious')
pl.legend(loc=0)
pl.title('SI epidemic without births or deaths, beta = 100 x gamma')
pl.xlabel('Time')
pl.ylabel('Susceptibles and Infectious')
pl.savefig('si1.png', dpi=900) # This does increase the resolution.
pl.show()
pd.DataFrame({'Days': t_range, 'Susceptible': RES[:,0], 'Infected': RES[:,1]}).to_csv("sis1.csv", index=False)



""" Model 2"""
gamma = 1/100
def diff_eqs(INP,t):
    
    Y=np.zeros((2))
    V = INP
    Y[0] = - beta * V[0] * V[1] + birth-death*V[0]
    Y[1] = beta * V[0] * V[1] - gamma * V[1]-death*V[1]
    return Y   # For odeint
t_start = 0.0; t_end = ND; t_inc = TS
t_range = np.arange(t_start, t_end+t_inc, t_inc)
RES = spi.odeint(diff_eqs,INPUT,t_range)
#print(RES)
#Ploting
pl.plot(RES[:,0], '-bs', label='Susceptibles')
pl.plot(RES[:,1], '-ro', label='Infectious')
pl.legend(loc=0)
pl.title('SI epidemic without births or deaths, beta = 100 x gamma')
pl.xlabel('Time')
pl.ylabel('Susceptibles and Infectious')
pl.savefig('si2.png', dpi=900) # This does increase the resolution.
pl.show()
pd.DataFrame({'Days': t_range, 'Susceptible': RES[:,0], 'Infected': RES[:,1]}).to_csv("sis2.csv", index=False)





""" Model 3"""
gamma = 1/10
def diff_eqs(INP,t):
    
    Y=np.zeros((2))
    V = INP
    Y[0] = - beta * V[0] * V[1] + birth-death*V[0]
    Y[1] = beta * V[0] * V[1] - gamma * V[1]-death*V[1]
    return Y   # For odeint
t_start = 0.0; t_end = ND; t_inc = TS
t_range = np.arange(t_start, t_end+t_inc, t_inc)
RES = spi.odeint(diff_eqs,INPUT,t_range)
#print(RES)
#Ploting
pl.plot(RES[:,0], '-bs', label='Susceptibles')
pl.plot(RES[:,1], '-ro', label='Infectious')
pl.legend(loc=0)
pl.title('SI epidemic without births or deaths, beta = 100 x gamma')
pl.xlabel('Time')
pl.ylabel('Susceptibles and Infectious')
pl.savefig('si3.png', dpi=900) # This does increase the resolution.
pl.show()
pd.DataFrame({'Days': t_range, 'Susceptible': RES[:,0], 'Infected': RES[:,1]}).to_csv("sis3.csv", index=False)







