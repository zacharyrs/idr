import math

import numpy

from idr.utility import py_cdf
from idr.inv_cdf import cdf, cdf_i, cdf_d1


def _FD_d1(x, mu, sigma, lamda):
    return (py_cdf(x+1e-6, mu, sigma, lamda) - py_cdf(x-1e-6, mu, sigma, lamda))/2e-6


def _py_cdf_d1(x, mu, sigma, lamda):
    pi = 3.14159265358979323846264338327950288419716939937510582
    pre = 1./math.sqrt(2*pi)
    noise = (1-lamda)*math.exp(-0.5*(x**2))
    norm_x = (x - mu)/sigma
    signal = lamda*math.exp(-0.5*(norm_x**2))
    return pre*(signal + noise)


def _py_cdf_d1_simple(x, mu, sigma, lamda):
    pi = 3.14159265358979323846264338327950288419716939937510582
    return -math.sqrt(2)*lamda*math.exp(-x**2/2)/(2*math.sqrt(pi)) \
        + math.sqrt(2)*lamda*math.exp(-(mu - x)**2/(2*sigma**2))/(2*math.sqrt(pi)*sigma) \
        + math.sqrt(2)*math.exp(-x**2/2)/(2*math.sqrt(pi))


def test_deriv():
    for x in range(10):
        fd = _FD_d1(x, 0, 1, 0.5)
        assert abs(_py_cdf_d1(x, 0, 1, 0.5) - fd) < 1e-5
        assert abs(_py_cdf_d1_simple(x, 0, 1, 0.5) - fd) < 1e-5
        assert abs(cdf_d1(x, 0, 1, 0.5) - fd) < 1e-5


def test_cdf_inverse():
    mu, sigma, lamda = 1.0, 1.0, 0.9
    lb, ub = -10.0, 10.0
    for r in [0.05, 0.1, 0.5, 0.9, 0.95]:
        x = cdf_i(r, mu, sigma, lamda, lb, ub, 1e-12)
        assert abs(cdf(x, mu, sigma, lamda) - r) < 1e-6
