#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# This file is created automatically from org-mode [[roam:Windkessel Python Module]]

"""
Module for 2, 3, and 4 Element Windkessel Models

Calculates pressures before - p[y] - and after R1 - p[1]

usage:

import windkessel as wp

solve_wk(fun, Rc, Rp, C, L, time_start, time_end, N, method, initial_cond, rtol)

where:

fun:                    function object - volume flow

R1, R2, C, L:           windkessel parameters
                        set L = 0 for 3-Element WK
                        set R1 = 0 for 2-Element WK

Units:  either use physiological units (ml/s, mmHg/ml/s, ...)
        or SI units (m^3, Pa/m^3/s, ...), the result will be in
        consistent units (mmHg, or Pa)


Optional:

time_start, time_end:   start and end time for integration
                        (float, default: 0, 10)

N:                      number of time steps for result
                        (integer, default: 10000)

method:                 integration method (string, default: "RK45")
                        see solve_ivp documentation for available methods

initial_cond:           initial conditions
                        (2 element list for serial wk              default: [0, 0])
                        (3 element list for parallel 4-element wk  default: [0, 0, 0])

rtol:                   relative tolerance (float, default 1e-6)

@author: Torsten Schenkel
"""

import numpy as np
import scipy as sp
import scipy.integrate


def ddt(fun, t, dt):
    # Central differencing method
    return (fun(t + dt) - fun(t - dt)) / (2 * dt)


def d2dt2(fun, t, dt):
    # Central differencing method
    return (fun(t + dt) - 2 * fun(t) + fun(t - dt)) / (dt ** 2)


def wk_old(t, y, Q, Rc, Rp, C, L, dt):
    # The 4-Element WK with serial L
    # set L = 0 for 3EWK
    # set L = 0, Rc = 0 for 2EWK
    #
    # Pressure at inlet:

    p1 = y[0]
    p2 = y[1]

    dp1dt = (
        Rc * ddt(Q, t, dt)
        + (1 + Rc / Rp) * Q(t) / C
        - p1 / (Rp * C)
        + L / (Rp * C) * ddt(Q, t, dt)
        + L * d2dt2(Q, t, dt)
    )
    # Pressure after proximal resistance:
    dp2dt = -p1 / (Rp * C) + (1 + Rc / Rp) * Q(t) / C + L / (Rp * C) * ddt(Q, t, dt)

    return [dp1dt, dp2dt]


def wk(t, y, Q, Rc, Rp, C, L, dt):
    # The 4-Element WK with serial L
    # set L = 0 for 3EWK
    # set L = 0, Rc = 0 for 2EWK
    #
    # Pressure at inlet:

    p1 = y[0]
    p2 = y[1]

    # Do not solve an ODE for p1!
    # This is calculated algebraically later!
    dp1dt = 0.0

    # Pressure after proximal resistance:
    dp2dt = Q(t) / C - p2 / (C * Rp)

    return [dp1dt, dp2dt]


def wk4p(t, y, Q, Rc, Rp, C, L, dt):
    # The 4-Element WK with parallel L

    p1 = y[0]
    p2 = y[1]

    dp1dt = y[2]
    d2pdt2 = (
        Rc * d2dt2(Q, t, dt)
        + (Rc / Rp + 1 / C) * ddt(Q, t, dt)
        + Rc / (C * L) * Q(t)
        - Rc / (Rp * C * L) * p1
        - (1 / (Rp * C) + Rc / L) * dp1dt
    )
    # Second pressure not implemented yet!
    dp2dt = 0

    return [dp1dt, dp2dt, d2pdt2]


def wk5(t, y, Q, Rc, Rp, C, Lp, Ls, dt):
    # The 4/5-Element WK with parallel Lp and serial Ls
    # changed from Ian's serial inertance being Lp in the Matlab code!
    # d_WM4E(1)  = -R1/L*WM4E(1) + (R1/L - 1/R2/C )*WM4E(2) + R1*(1+Lp/L)*didt + i/C;
    # d_WM4E(2)  =               - 1/R2/C          *WM4E(2)                    + i/C;

    dp1dt = (
        -Rc / Lp * y[0]
        + (Rc / Lp - 1 / Rp / C) * y[1]
        + Rc * (1 + Ls / Lp) * ddt(Q, t, dt)
        + Q(t) / C
    )

    dp2dt = -1 / Rp / C * y[1] + Q(t) / C

    return [dp1dt, dp2dt]


def solve_wk(
    fun,
    Rc,
    Rp,
    C,
    L=0,
    time_start=0,
    time_end=10,
    N=10000,
    method="RK45",
    initial_cond=[0, 0],
    rtol=1e-6,
    dt=0.01,
):

    result = sp.integrate.solve_ivp(
        lambda t, y: wk(t, y, fun, Rc, Rp, C, L, dt),
        (time_start, time_end),
        initial_cond,
        t_eval=np.linspace(time_start, time_end, N),
        method=method,
        rtol=rtol,
        vectorized=True,
    )

    # p1 is not calculated from an ODE, but algebraically as:
    result.y[0] = result.y[1] + L * ddt(fun, result.t, dt) + Rc * fun(result.t)

    return result


def solve_wk4p(
    fun,
    Rc,
    Rp,
    C,
    L,
    time_start=0,
    time_end=10,
    N=10000,
    method="RK45",
    initial_cond=[0, 0, 0],
    rtol=1e-6,
    dt=0.01,
):

    return sp.integrate.solve_ivp(
        lambda t, y: wk4p(t, y, fun, Rc, Rp, C, L, dt),
        (time_start, time_end),
        initial_cond,
        t_eval=np.linspace(time_start, time_end, N),
        method=method,
        rtol=rtol,
        vectorized=True,
    )


def solve_wk5(
    fun,
    Rc,
    Rp,
    C,
    Lp,
    Ls=0,
    time_start=0,
    time_end=10,
    N=10000,
    method="RK45",
    initial_cond=[0, 0, 0],
    rtol=1e-6,
    dt=0.01,
):

    return sp.integrate.solve_ivp(
        lambda t, y: wk5(t, y, fun, Rc, Rp, C, Lp, Ls, dt),
        (time_start, time_end),
        initial_cond,
        t_eval=np.linspace(time_start, time_end, N),
        method=method,
        rtol=rtol,
        vectorized=True,
    )


def nearest_time_index(t_array, t):

    t = t % np.max(t_array)

    idx = (np.abs(t_array - t)).argmin()

    return idx


def low_time_index(t_array, t):

    t = t % np.max(t_array)

    idx = nearest_time_index(t_array, t)

    if t_array[idx] > t:
        idx = idx - 1

    return idx


def local_PWBezier(data, t):
    """
    local approximation of values for piecewise cubic Bezier approximator
    """

    t = t % np.max(data[0])

    n = low_time_index(data[0], t)

    # Extend arrays by 1 (roll-over) for n+2 index!
    # time = np.append[data[0], data[0][0]]
    # data[1] = np.append[data[1], data[1][0]]

    # relative time between the reference points:
    intime = (t - data[0][n]) / (data[0][n + 1] - data[0][n])

    b0 = 1.0 / 6.0 * (data[1][n - 1] + 4.0 * data[1][n] + data[1][n + 1])
    b1 = 1.0 / 3.0 * (2 * data[1][n] + data[1][n + 1])
    b2 = 1.0 / 3.0 * (data[1][n] + 2 * data[1][n + 1])
    # last term includes rollover to -1 for n + 2
    if n < len(data[0]) - 2:
        b3 = 1.0 / 6.0 * (data[1][n] + 4 * data[1][n + 1] + data[1][n + 2])
    else:
        b3 = 1.0 / 6.0 * (data[1][n] + 4 * data[1][n + 1] + data[1][0])

    # b0 = 1.0 / 6.0 * (data[1][n - 2] + 4.0 * data[1][n - 1] + data[1][n])
    # b1 = 1.0 / 3.0 * (2 * data[1][n - 1] + data[1][n])
    # b2 = 1.0 / 3.0 * (data[1][n - 1] + 2 * data[1][n])
    # b3 = 1.0 / 6.0 * (data[1][n - 1] + 4 * data[1][n] + data[1][n + 1])

    return (
        pow((1.0 - intime), 3.0) * b0
        + 3.0 * pow((1.0 - intime), 2.0) * intime * b1
        + 3.0 * (1.0 - intime) * pow(intime, 2.0) * b2
        + pow(intime, 3.0) * b3
    )


def PWBezier(data, t):
    '''
    Piecewise cubic Bezier curve approximator
    '''

    # Check if t is a float, if yes, call local approximator,
    # if not denumerate time array and create array of local approximations
    if isinstance(t, float):
        return local_PWBezier(data, t)
    else:
        v = np.zeros(t.shape)
        for idx, time in np.ndenumerate(t):
            v[idx[0]] = local_PWBezier(data, time)
        return v
