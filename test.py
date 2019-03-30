#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 21:28:01 2019

@author: sermetcagan
"""

from __future__ import print_function
from IPython.display import display, clear_output
import sys, os
import time
import math
import getdist.plots as gplot
from sympy import *
from sympy.solvers import solve
from scipy import optimize
from scipy.constants import physical_constants
from sympy.physics.mechanics import dynamicsymbols, init_vprinting
from matplotlib import pyplot as plt
import scipy.integrate as integrate
import numpy as np

init_vprinting()
x_prime = Function('x^{\prime}')
y_prime = Function('y^{\prime}')

l = Symbol('\lambda')
x, y, w_i = symbols('x y w_{i}')

x_prime = sqrt(3/2)*l*y**2 + 3/2 * x *((1-w_i)*(x**2 - 1) - y**2 * (1 + w_i))
y_prime = -1 * sqrt(3/2)*l*x*y + 3/2 * y * (x**2 * (1 - w_i) + (1 + w_i)*(1-y**2))
solve([x_prime + y_prime], x, y)