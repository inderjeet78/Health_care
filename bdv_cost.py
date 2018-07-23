#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 22:27:05 2018

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
vac = 1/1000
S0=0.8
I0=0.1
R0=1-S0-I0
INPUT = (S0, I0, R0)
def diff_eqs(INPUT,t):  
	'''The main set of equations'''
	Y=np.zeros((3))
	V = INPUT    
	Y[0] = birth*(V[0]+V[1]+V[2]) - beta * V[0] * V[1] - death * V[0] - vac*V[0]
	Y[1] = beta * V[0] * V[1] - gamma * V[1] - 2*death * V[1]
	Y[2] = gamma * V[1] - death * V[2]+ vac*V[0]
	return Y
time= np.arange(0,366,1)
RES = spi.odeint(diff_eqs,INPUT,time)
#print ((RES).shape)
#Ploting
pl.subplot(311)
pl.plot(RES[:,0], '-y', label='Susceptibles')
pl.title('SIR Model with Birth, Death and vaccination rate cosidered and beta = 100 x gamma')
pl.xlabel('Time')
pl.ylabel('Susceptibles')
pl.subplot(312)
pl.plot(RES[:,1], '-r', label='Infectious')
pl.xlabel('Time')
pl.ylabel('Infectious')
pl.subplot(313)
pl.plot(RES[:,2], '-g', label='Recovereds')
pl.xlabel('Time')
pl.ylabel('Recovereds')
pl.savefig('bdc11.png', dpi=900)
pl.show()
fig = plt.figure(facecolor='w')
ax = fig.add_subplot(111, axis_bgcolor='white', axisbelow=True)
plt.title('SIR Model with Birth, Death and vaccination rate cosidered and beta = 100 x gamma')
ax.plot(time,RES[:,0], 'y', alpha=0.5, lw=2, label='Susceptible')
ax.plot(time,RES[:,1], 'r', alpha=0.5, lw=2, label='Infected')
ax.plot(time,RES[:,2], 'g', alpha=0.5, lw=2, label='Recovered with immunity')
ax.set_xlabel('Time ')
ax.set_ylabel('Fraction of population')
#ax.set_ylim(0,.2)
#ax.set_xlim(0,.2)
ax.yaxis.set_tick_params(length=0)
ax.xaxis.set_tick_params(length=0)
ax.grid(b=True, which='major', c='w', lw=2, ls='-')
legend = ax.legend()
legend.get_frame().set_alpha(0.5)
for spine in ('top', 'right', 'bottom', 'left'):
    ax.spines[spine].set_visible(False)
plt.savefig('bdc12.png', dpi=900)
plt.show()
#  Amount Spent on Susceptible people due to vaccination
""" Let Size of population is 100000"""
P = 100000
vc=10
recc=1000
vac_cost = np.zeros(366)
vac_cost_cum=np.zeros(366)
""" Calculation of Vaccination cost """
for i in range(366):
    vac_cost[i] = (RES[i,0]*P*vc*vac)
    
vac_cost_cum[0]=vac_cost[0]
for i in range(1,366):
    vac_cost_cum[i] = vac_cost_cum[i-1] +vac_cost[i]
#print ((vac_cost))
#print(vac_cost_cum)
fig = plt.figure(facecolor='w')
ax = fig.add_subplot(111, axis_bgcolor='#dddddddd', axisbelow=True)
ax.plot(time,vac_cost_cum, 'y', alpha=1, lw=2, label='Vaccination Cost')
ax.set_ylabel('Vaccination Cost per day ')
ax.set_xlabel('Days')
ax.yaxis.set_tick_params(length=0)
ax.xaxis.set_tick_params(length=0)
ax.grid(b=True, which='major', c='w', lw=2, ls='-')
legend = ax.legend()
legend.get_frame().set_alpha(0.5)
for spine in ('top', 'right', 'bottom', 'left'):
    ax.spines[spine].set_visible(False)
plt.savefig('bdc13.png', dpi=900)
plt.show()
""" Calculation of treatment cost"""
rec_cost = np.zeros(366)
rec_cost_cum = np.zeros(366)

for i in range(366):
    if i==0:
        rec_cost[i]=0
    else:
        rec_cost[i] = ((RES[i,2]-RES[i-1,2]-RES[i,0]*vac)*P*recc)
rec_cost_cum[0]=rec_cost[0]
for i in range(1,366):
    rec_cost_cum[i] = rec_cost_cum[i-1] +rec_cost[i]    
#print (sum(rec_cost))
fig = plt.figure(facecolor='w')
ax = fig.add_subplot(111, axis_bgcolor='#dddddd', axisbelow=True)
ax.plot(time,rec_cost_cum, 'r', alpha=1, lw=2, label='Treatement Cost')
ax.set_ylabel('Treatment Cost per day ')
ax.set_xlabel('Days')
ax.yaxis.set_tick_params(length=0)
ax.xaxis.set_tick_params(length=0)
ax.grid(b=True, which='major', c='w', lw=2, ls='-')
legend = ax.legend()
legend.get_frame().set_alpha(0.5)
for spine in ('top', 'right', 'bottom', 'left'):
    ax.spines[spine].set_visible(False)
plt.savefig('bdc14.png', dpi=900)
plt.show()
print ("Total Cost spent on vaccination in 1 year :",sum(vac_cost))
print ("Total Cost spent on Treatement in 1 year :",sum(rec_cost))
pd.DataFrame({'Days': time, 'Susceptible': RES[:,0], 'Infected': RES[:,1], 'Recovered': RES[:,2]}).to_csv("bdv_cost1.csv", index=False)


""" Model 2"""

gamma = 1/20
RES = spi.odeint(diff_eqs,INPUT,time)
#print ((RES).shape)
#Ploting
pl.subplot(311)
pl.plot(RES[:,0], '-y', label='Susceptibles')
pl.title('SIR Model with Birth, Death and vaccination rate cosidered and beta = 2 x gamma')
pl.xlabel('Time')
pl.ylabel('Susceptibles')
pl.subplot(312)
pl.plot(RES[:,1], '-r', label='Infectious')
pl.xlabel('Time')
pl.ylabel('Infectious')
pl.subplot(313)
pl.plot(RES[:,2], '-g', label='Recovereds')
pl.xlabel('Time')
pl.ylabel('Recovereds')
pl.savefig('bdc21.png', dpi=900)
pl.show()
fig = plt.figure(facecolor='w')
ax = fig.add_subplot(111, axis_bgcolor='white', axisbelow=True)
plt.title('SIR Model with Birth, Death and vaccination rate cosidered and beta = 2 x gamma')
ax.plot(time,RES[:,0], 'y', alpha=0.5, lw=2, label='Susceptible')
ax.plot(time,RES[:,1], 'r', alpha=0.5, lw=2, label='Infected')
ax.plot(time,RES[:,2], 'g', alpha=0.5, lw=2, label='Recovered with immunity')
ax.set_xlabel('Time ')
ax.set_ylabel('Fraction of population')
#ax.set_ylim(0,.2)
#ax.set_xlim(0,.2)
ax.yaxis.set_tick_params(length=0)
ax.xaxis.set_tick_params(length=0)
ax.grid(b=True, which='major', c='w', lw=2, ls='-')
legend = ax.legend()
legend.get_frame().set_alpha(0.5)
for spine in ('top', 'right', 'bottom', 'left'):
    ax.spines[spine].set_visible(False)
plt.savefig('bdc22.png', dpi=900)
plt.show()
#  Amount Spent on Susceptible people due to vaccination
""" Let Size of population is 100000"""
P = 100000
vc=10
recc=1000
vac_cost = np.zeros(366)
vac_cost_cum=np.zeros(366)
""" Calculation of Vaccination cost """
for i in range(366):
    vac_cost[i] = (RES[i,0]*P*vc*vac)
    
vac_cost_cum[0]=vac_cost[0]
for i in range(1,366):
    vac_cost_cum[i] = vac_cost_cum[i-1] +vac_cost[i]
#print ((vac_cost))
#print(vac_cost_cum)
fig = plt.figure(facecolor='w')
ax = fig.add_subplot(111, axis_bgcolor='#dddddddd', axisbelow=True)
ax.plot(time,vac_cost_cum, 'y', alpha=1, lw=2, label='Vaccination Cost')
ax.set_ylabel('Vaccination Cost per day ')
ax.set_xlabel('Days')
ax.yaxis.set_tick_params(length=0)
ax.xaxis.set_tick_params(length=0)
ax.grid(b=True, which='major', c='w', lw=2, ls='-')
legend = ax.legend()
legend.get_frame().set_alpha(0.5)
for spine in ('top', 'right', 'bottom', 'left'):
    ax.spines[spine].set_visible(False)
plt.savefig('bdc23.png', dpi=900)
plt.show()
""" Calculation of treatment cost"""
rec_cost = np.zeros(366)
rec_cost_cum = np.zeros(366)

for i in range(366):
    if i==0:
        rec_cost[i]=0
    else:
        rec_cost[i] = ((RES[i,2]-RES[i-1,2]-RES[i,0]*vac)*P*recc)
rec_cost_cum[0]=rec_cost[0]
for i in range(1,366):
    rec_cost_cum[i] = rec_cost_cum[i-1] +rec_cost[i]    
#print (sum(rec_cost))
fig = plt.figure(facecolor='w')
ax = fig.add_subplot(111, axis_bgcolor='#dddddd', axisbelow=True)
ax.plot(time,rec_cost_cum, 'r', alpha=1, lw=2, label='Treatement Cost')
ax.set_ylabel('Treatment Cost per day ')
ax.set_xlabel('Days')
ax.yaxis.set_tick_params(length=0)
ax.xaxis.set_tick_params(length=0)
ax.grid(b=True, which='major', c='w', lw=2, ls='-')
legend = ax.legend()
legend.get_frame().set_alpha(0.5)
for spine in ('top', 'right', 'bottom', 'left'):
    ax.spines[spine].set_visible(False)
plt.savefig('bdc24.png', dpi=900)
plt.show()
print ("Total Cost spent on vaccination in 1 year :",sum(vac_cost))
print ("Total Cost spent on Treatement in 1 year :",sum(rec_cost))
pd.DataFrame({'Days': time, 'Susceptible': RES[:,0], 'Infected': RES[:,1], 'Recovered': RES[:,2]}).to_csv("bdv_cost2.csv", index=False)



""" Model 3"""


gamma = 1/10
RES = spi.odeint(diff_eqs,INPUT,time)
#print ((RES).shape)
#Ploting
pl.subplot(311)
pl.plot(RES[:,0], '-y', label='Susceptibles')
pl.title('SIR Model with Birth, Death and vaccination rate cosidered and beta = gamma')
pl.xlabel('Time')
pl.ylabel('Susceptibles')
pl.subplot(312)
pl.plot(RES[:,1], '-r', label='Infectious')
pl.xlabel('Time')
pl.ylabel('Infectious')
pl.subplot(313)
pl.plot(RES[:,2], '-g', label='Recovereds')
pl.xlabel('Time')
pl.ylabel('Recovereds')
pl.savefig('bdc31.png', dpi=900)
pl.show()
fig = plt.figure(facecolor='w')
ax = fig.add_subplot(111, axis_bgcolor='white', axisbelow=True)
plt.title('SIR Model with Birth, Death and vaccination rate cosidered and beta = gamma')
ax.plot(time,RES[:,0], 'y', alpha=0.5, lw=2, label='Susceptible')
ax.plot(time,RES[:,1], 'r', alpha=0.5, lw=2, label='Infected')
ax.plot(time,RES[:,2], 'g', alpha=0.5, lw=2, label='Recovered with immunity')
ax.set_xlabel('Time ')
ax.set_ylabel('Fraction of population')
#ax.set_ylim(0,.2)
#ax.set_xlim(0,.2)
ax.yaxis.set_tick_params(length=0)
ax.xaxis.set_tick_params(length=0)
ax.grid(b=True, which='major', c='w', lw=2, ls='-')
legend = ax.legend()
legend.get_frame().set_alpha(0.5)
for spine in ('top', 'right', 'bottom', 'left'):
    ax.spines[spine].set_visible(False)
plt.savefig('bdc32.png', dpi=900)
plt.show()
#  Amount Spent on Susceptible people due to vaccination
""" Let Size of population is 100000"""
P = 100000
vc=10
recc=1000
vac_cost = np.zeros(366)
vac_cost_cum=np.zeros(366)
""" Calculation of Vaccination cost """
for i in range(366):
    vac_cost[i] = (RES[i,0]*P*vc*vac)
    
vac_cost_cum[0]=vac_cost[0]
for i in range(1,366):
    vac_cost_cum[i] = vac_cost_cum[i-1] +vac_cost[i]
#print ((vac_cost))
#print(vac_cost_cum)
fig = plt.figure(facecolor='w')
ax = fig.add_subplot(111, axis_bgcolor='#dddddddd', axisbelow=True)
ax.plot(time,vac_cost_cum, 'y', alpha=1, lw=2, label='Vaccination Cost')
ax.set_ylabel('Vaccination Cost per day ')
ax.set_xlabel('Days')
ax.yaxis.set_tick_params(length=0)
ax.xaxis.set_tick_params(length=0)
ax.grid(b=True, which='major', c='w', lw=2, ls='-')
legend = ax.legend()
legend.get_frame().set_alpha(0.5)
for spine in ('top', 'right', 'bottom', 'left'):
    ax.spines[spine].set_visible(False)
plt.savefig('bdc33.png', dpi=900)
plt.show()
""" Calculation of treatment cost"""
rec_cost = np.zeros(366)
rec_cost_cum = np.zeros(366)

for i in range(366):
    if i==0:
        rec_cost[i]=0
    else:
        rec_cost[i] = ((RES[i,2]-RES[i-1,2]-RES[i,0]*vac)*P*recc)
rec_cost_cum[0]=rec_cost[0]
for i in range(1,366):
    rec_cost_cum[i] = rec_cost_cum[i-1] +rec_cost[i]    
#print (sum(rec_cost))
fig = plt.figure(facecolor='w')
ax = fig.add_subplot(111, axis_bgcolor='#dddddd', axisbelow=True)
ax.plot(time,rec_cost_cum, 'r', alpha=1, lw=2, label='Treatement Cost')
ax.set_ylabel('Treatment Cost per day ')
ax.set_xlabel('Days')
ax.yaxis.set_tick_params(length=0)
ax.xaxis.set_tick_params(length=0)
ax.grid(b=True, which='major', c='w', lw=2, ls='-')
legend = ax.legend()
legend.get_frame().set_alpha(0.5)
for spine in ('top', 'right', 'bottom', 'left'):
    ax.spines[spine].set_visible(False)
plt.savefig('bdc34.png', dpi=900)
plt.show()
print ("Total Cost spent on vaccination in 1 year :",sum(vac_cost))
print ("Total Cost spent on Treatement in 1 year :",sum(rec_cost))
pd.DataFrame({'Days': time, 'Susceptible': RES[:,0], 'Infected': RES[:,1], 'Recovered': RES[:,2]}).to_csv("bdv_cost3.csv", index=False)


