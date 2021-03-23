#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module for 2, 3, and 4 Element Windkessel Models

Calculates pressures before - p[y] - and after R1 - p[1]

usage:
    
import windkessel as wp

solve_wk(I, R1, R2, C, L, time_start, time_end, N, method, initial_cond, rtol)

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

time_start, time_end:   start and end time for integration
                        (float, default: 0, 10)
                        
N:                      number of time steps for result
                        (integer, default: 10000)

method:                 integration method (string, default: "RK45")
                        see solve_ivp documentation for available methods 

initial_cond:           initial conditions (2 element list, default: [0, 0])

rtol:                   relative tolerance (float, default 1e-6)


Created on Sun Mar 21 22:04:43 2021

@author: acests3
"""
import numpy as np
import scipy as sp
import scipy.integrate


def ddt(I, t, dt):
    # Central differencing method
    return (I(t + dt) - I(t - dt)) / (2 * dt)


def d2dt2(I, t, dt):
    # Central differencing method
    return (I(t + dt) - 2 * I(t) + I(t - dt)) / (dt ** 2)


def wk(t, p, I, R1, R2, C, L, dt):
    dp = np.zeros(2)
    # The 4-Element WK
    # set L = 0 for 3EWK
    # set L = 0, R1 = 0 for 2EWK
    #
    # Pressure at inlet:
    dp[0] = (
        R1 * ddt(I, t, dt)
        + (1 + R1 / R2) * I(t) / C
        - p[0] / (R2 * C)
        + L / (R2 * C) * ddt(I, t, dt)
        + L * d2dt2(I, t, dt)
    )
    # Pressure after proximal resistance:
    dp[1] = (
        -p[0] / (R2 * C)
        + (1 + R1 / R2) * I(t) / C
        + L / (R2 * C) * ddt(I, t, dt)
        # Alternative formulation
        # dp[0]
        # - R1 * ddt(I, t)
        # - L * d2dt2(I, t)
    )

    return dp


def wk4p(t, p, I, R1, R2, C, L, dt):
    dp = np.zeros(2)
    # The 4-Element WK
    # set L = 0 for 3EWK
    # set L = 0, R1 = 0 for 2EWK
    #
    # Pressure at inlet:

    dp[0] = p[1]
    dp[1] = (
        R1 * d2dt2(I, t, dt)
        + (R1 / R2 + I(t) / C) * ddt(I, t, dt)
        - R1 / (R2 * C * L) * p[0]
        - (1 / (R2 * C) + R1 / C) * p[1]
    )
    # pressure after proximal resistance
    # dp[2] = R1 / L * (p[2] - p[1]) - R1 * ddt(I, t, dt) - dp[1]

    return dp


def solve_wk(
    I,
    R1,
    R2,
    C,
    L,
    time_start=0,
    time_end=10,
    N=10000,
    method="RK45",
    initial_cond=[0, 0],
    rtol=1e-6,
):

    dt = (time_end - time_start) / N / 10

    return sp.integrate.solve_ivp(
        lambda t, p: wk(t, p, I, R1, R2, C, L, dt),
        (time_start, time_end),
        initial_cond,
        t_eval=np.linspace(time_start, time_end, N),
        method=method,
        rtol=rtol,
        vectorized=True,
    )


def solve_wk4p(
    I,
    R1,
    R2,
    C,
    L,
    time_start=0,
    time_end=10,
    N=10000,
    method="RK45",
    initial_cond=[0, 0, 0],
    rtol=1e-6,
):

    dt = (time_end - time_start) / N / 10

    return sp.integrate.solve_ivp(
        lambda t, p: wk4p(t, p, I, R1, R2, C, L, dt),
        (time_start, time_end),
        initial_cond,
        t_eval=np.linspace(time_start, time_end, N),
        method=method,
        rtol=rtol,
        vectorized=True,
    )
