#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 19:16:15 2018

@author: inderjeetsingh
"""

import scipy.integrate as spi
import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
import pandas as pd
death=7.3/(1000*365.0)
birth=19.3/(1000*365.0)
beta=1/100
gamma=1/1000
S0=0.9
I0=0.00
R0=1-S0-I0
INPUT = (S0, I0, R0)

def diff_eqs(INPUT,t):  
	'''The main set of equations'''
	Y=np.zeros((3))
	V = INPUT    
	Y[0] = birth*(V[0]+V[1]+V[2]) - beta * V[0] * V[1] - death * V[0]
	Y[1] = beta * V[0] * V[1] - gamma * V[1] - 2*death * V[1]
	Y[2] = gamma * V[1] - death * V[2]
	return Y   # For odeint



time= np.arange(0,366,1)
RES = spi.odeint(diff_eqs,INPUT,time)

#print (RES)
#Ploting
pl.subplot(311)
pl.plot(RES[:,0], '-y', label='Susceptibles')
pl.title('SIR Model with Birth and Death rate cosidered')
pl.xlabel('Time')
pl.ylabel('Susceptibles')
pl.subplot(312)
pl.plot(RES[:,1], '-r', label='Infectious')
pl.xlabel('Time')
pl.ylabel('Infectious')
pl.subplot(313)
pl.plot(RES[:,2], '-g', label='Recovered')
pl.xlabel('Time')
pl.ylabel('Recovereds')
pl.savefig('bd11.png', dpi=900)
pl.show()



fig = plt.figure(facecolor='w')
ax = fig.add_subplot(111, axis_bgcolor='#dddddddd', axisbelow=True)
ax.plot(time,RES[:,0], 'y', alpha=1, lw=2, label='Susceptible')
ax.plot(time,RES[:,1], 'r', alpha=1, lw=2, label='Infected')
ax.plot(time,RES[:,2], 'g', alpha=1, lw=2, label='Recovered with immunity')
ax.set_xlabel('Time ')
ax.set_ylabel('Fraction of population w.r.t initial population')
#ax.set_ylim(0,.2)
#ax.set_xlim(0,.2)
ax.yaxis.set_tick_params(length=0)
ax.xaxis.set_tick_params(length=0)
ax.grid(b=True, which='major', c='w', lw=2, ls='-')
legend = ax.legend()
legend.get_frame().set_alpha(0.2)
for spine in ('top', 'right', 'bottom', 'left'):
    ax.spines[spine].set_visible(False)
plt.savefig('bg12.png', dpi=900)
plt.show()
pd.DataFrame({'Days': time, 'Susceptible': RES[:,0], 'Infected': RES[:,1], 'Recovered': RES[:,2]}).to_csv("birth_death1.csv", index=False)
