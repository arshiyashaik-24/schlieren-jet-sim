"""Exact solution of the 1D Euler Riemann problem (Toro, ch. 4).

Used as the reference for the shock-tube validation of the flow solver.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

GAMMA = 1.4


def _f(p, rho, pk, g):
    """Pressure function f_K(p) and the speed of sound of side K."""
    c = np.sqrt(g * pk / rho)
    if p > pk:  # shock
        A = 2.0 / ((g + 1) * rho)
        B = (g - 1) / (g + 1) * pk
        return (p - pk) * np.sqrt(A / (p + B)), c
    return 2 * c / (g - 1) * ((p / pk) ** ((g - 1) / (2 * g)) - 1), c  # rarefaction


def star_state(left, right, g=GAMMA):
    rl, ul, pl = left
    rr, ur, pr = right
    du = ur - ul
    p_star = brentq(lambda p: _f(p, rl, pl, g)[0] + _f(p, rr, pr, g)[0] + du, 1e-12, 100 * max(pl, pr))
    u_star = 0.5 * (ul + ur) + 0.5 * (_f(p_star, rr, pr, g)[0] - _f(p_star, rl, pl, g)[0])
    return p_star, u_star


def sample(left, right, xi, g=GAMMA):
    """Exact (rho, u, p) at similarity coordinates xi = x / t."""
    rl, ul, pl = left
    rr, ur, pr = right
    ps, us = star_state(left, right, g)
    cl = np.sqrt(g * pl / rl)
    cr = np.sqrt(g * pr / rr)
    gm = (g - 1) / (g + 1)
    out = np.empty((3, len(xi)))
    for i, s in enumerate(xi):
        if s <= us:  # left of contact
            if ps > pl:  # left shock
                rs = rl * (ps / pl + gm) / (gm * ps / pl + 1)
                S = ul - cl * np.sqrt((g + 1) / (2 * g) * ps / pl + (g - 1) / (2 * g))
                out[:, i] = (rl, ul, pl) if s < S else (rs, us, ps)
            else:  # left rarefaction
                rs = rl * (ps / pl) ** (1 / g)
                cs = cl * (ps / pl) ** ((g - 1) / (2 * g))
                if s < ul - cl:
                    out[:, i] = (rl, ul, pl)
                elif s > us - cs:
                    out[:, i] = (rs, us, ps)
                else:
                    u = 2 / (g + 1) * (cl + (g - 1) / 2 * ul + s)
                    c = 2 / (g + 1) * (cl + (g - 1) / 2 * (ul - s))
                    rho = rl * (c / cl) ** (2 / (g - 1))
                    out[:, i] = (rho, u, pl * (c / cl) ** (2 * g / (g - 1)))
        else:  # right of contact
            if ps > pr:  # right shock
                rs = rr * (ps / pr + gm) / (gm * ps / pr + 1)
                S = ur + cr * np.sqrt((g + 1) / (2 * g) * ps / pr + (g - 1) / (2 * g))
                out[:, i] = (rr, ur, pr) if s > S else (rs, us, ps)
            else:  # right rarefaction
                rs = rr * (ps / pr) ** (1 / g)
                cs = cr * (ps / pr) ** ((g - 1) / (2 * g))
                if s > ur + cr:
                    out[:, i] = (rr, ur, pr)
                elif s < us + cs:
                    out[:, i] = (rs, us, ps)
                else:
                    u = 2 / (g + 1) * (-cr + (g - 1) / 2 * ur + s)
                    c = 2 / (g + 1) * (cr - (g - 1) / 2 * (ur - s))
                    rho = rr * (c / cr) ** (2 / (g - 1))
                    out[:, i] = (rho, u, pr * (c / cr) ** (2 * g / (g - 1)))
    return out
