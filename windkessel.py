#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module for 2, 3, and 4 Element Windkessel Models

Calculates pressures before - p[y] - and after R1 - p[1]

usage:
    
import windkessel as wp

solveWK(I, R1, R2, C, L, timeStart, timeEnd, N, method, initialCond, rtol)

where:
    
I:                      function object I(t) - volume flow
                        
R1, R2, C, L:           windkessel parameters
                        set L = 0 for 3-Element WK
                        set R1 = 0 for 2-Element WK
                        
Units:  either use physiological units (ml/s, mmHg/ml/s, ...) 
        or SI units (m^3, Pa/m^3/s, ...), the result will be in
        consistent units (mmHg, or Pa)
         
               
Optional (use commas to give default option in case a subsequent option is
          to be given):        

timeStart, timeEnd:     start and end time for integration
                        (float, default: 0, 10)
                        
N:                      number of time steps for result
                        (integer, default: 10000)

method:                 integration method (string, default: "RK45")
                        see solve_ivp documentation for available methods 

initialCond:            initial conditions (2 element list, default: [0, 0])

rtol:                   relative tolerance (float, default 1e-6)


Created on Sun Mar 21 22:04:43 2021

@author: acests3
"""
import numpy as np
import scipy as sp
import scipy.integrate

dt1 = 1e-6
dt2 = 1.1 * dt1

def ddt(I, t):
    return (I(t + dt1) - I(t - dt1)) / 2 / dt1


def d2dt2(I, t):
    return (
        ((I(t + dt1 + dt2) - I(t + dt1 -dt2)) / 2 / dt2
        - (I(t - dt1 + dt2) - I(t - dt1 -dt2)) / 2 / dt2)
        / 2 / dt1
    )

def WK(t, p, I, R1, R2, C, L):
    dp = np.zeros(2)
    # The 4-Element WK
    # set L = 0 for 3EWK
    # set L = 0, R1 = 0 for 2EWK
    
    # Pressure at inlet:
    dp[0] = (
        R1 * ddt(I, t)
        + (1 + R1 / R2) * I(t) / C
        - p[0] / (R2 * C)
        + L / (R2 * C) * ddt(I, t)
        + L * d2dt2(I, t)
    )
    # Pressure after proximal resistance:
    dp[1] = (
        # R1 * ddt(I, t)
        -p[0] / (R2 * C) 
        + (1 + R1 / R2) * I(t) / C
        # + L / (R2 * C) * ddt(I, t)
        # + L * d2dt2(I, t) 
    )

    return dp

def solveWK(I, R1, R2, C, L, 
            timeStart = 0, timeEnd = 10, N = 10000, 
            method = "RK45", initialCond = [0, 0], rtol = 1e-6):
     
    return sp.integrate.solve_ivp(
        lambda t,p: WK(t, p, I, R1, R2, C, L),
        (timeStart, timeEnd),
        initialCond,
        t_eval = np.linspace(timeStart, timeEnd, N),
        method = method,
        rtol = rtol,
        vectorized = True
    )  

