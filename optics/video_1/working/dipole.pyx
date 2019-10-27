import numpy as np
# cimport numpy as np
from scipy.misc import derivative
from scipy.optimize import root_scalar

import cython

def get_retarded_time(double x,
                      double z,
                      q_x_func,
                      q_z_func,
                      double t,
                      double C):
    def func(t_r):
        return np.sqrt((x - q_x_func(t_r)) ** 2 + (z - q_z_func(t_r)) ** 2) - C * (t - t_r)

    result = root_scalar(func, x0=0, x1=1)

    return result.root

# pos_x from other script
# pos_z from other script

def E_at_point(double x,
               double z,
               double t,
               q_x_func=None,
               q_z_func=None,
               double C=0,
               double EPS_0=0,
               double Q=0):
    """
    Počítá komponenty X a Z pole pohybujícího se náboje v bodě [x, z]
    Parameters
    ----------
    x : float
        Pole hodnot X mřížky
    z : float
        Pole hodnot Y mřížky
    t : float
        Čas
    q_x_func : callable
        Funkce souřadnice X bodového náboje v čase
    q_z_func : callable
        Funkce souřadnice Z bodového náboje v čase
    C : float
        Rychlost světla
    EPS_0 : float
        Permitivita vakua
    Q : float
        Velikost náboje

    Returns
    -------
    E_x : float
        Xová složka E
    E_z : float
        Zová složka E
    """
    if not q_x_func:
        q_x_func = lambda t: 0
    if not q_z_func:
        q_z_func = lambda t: 0
    if not C:
        C = 2
    if not EPS_0:
        EPS_0 = 1
    if not Q:
        Q = 1

    t_ret = get_retarded_time(x, z, q_x_func, q_z_func, t, C)
    cdef double nice_r_x = x - q_x_func(t_ret)
    cdef double nice_r_z = z - q_z_func(t_ret)

    cdef double nice_r_norm = np.sqrt(nice_r_x ** 2 + nice_r_z ** 2)

    cdef double v_x_ret = derivative(q_x_func, t_ret, dx=1e-6)
    cdef double v_z_ret = derivative(q_z_func, t_ret, dx=1e-6)
    cdef double a_x_ret = derivative(q_x_func, t_ret, dx=1e-6, n=2)
    cdef double a_z_ret = derivative(q_z_func, t_ret, dx=1e-6, n=2)

    cdef double u_x = C * nice_r_x / nice_r_norm - v_x_ret
    cdef double u_z = C * nice_r_z / nice_r_norm - v_z_ret

    cdef double nice_r_dot_u = nice_r_x * u_x + nice_r_z * u_z
    cdef double const = Q / (4 * np.pi * EPS_0)
    cdef double front = nice_r_norm / (nice_r_dot_u ** 3)

    cdef double radiation_term_x = -nice_r_z * (u_z * a_x_ret - u_x * a_z_ret)
    cdef double radiation_term_z = nice_r_x * (u_z * a_x_ret - u_x * a_z_ret)

    E_x = const * front * radiation_term_x
    E_z = const * front * radiation_term_z

    return E_x, E_z

# @cython.embedsignature(True)
# nefunguje jak má
def E_components_on_grid(xs, zs, t,
                         q_x_func=None, q_z_func=None,
                         C=0, EPS_0=0, Q=0,
                         x_origin=0, z_origin=0, t_origin=0,
                         mask_func=None):
    """
    Počítá komponenty X a Z pole pohybujícího se náboje na mřížce bodů
    Parameters
    ----------
    xs : sorted iterable of floats
        Pole hodnot X mřížky
    zs : sorted iterable of floats
        Pole hodnot Y mřížky
    t : float
        Čas
    q_x_func : callable
        Funkce souřadnice X bodového náboje v čase
    q_z_func : callable
        Funkce souřadnice Z bodového náboje v čase
    C : float
        Rychlost světla
    EPS_0 : float
        Permitivita vakua
    Q : float
        Velikost náboje
    x_origin : float
        Posunutí počátku souřadnice X (aby bylo možné použít stejnou funkci
        polohy náboje na čase pro náboje v různých místech)
    z_origin : float
        Posunutí počátku souřadnice Z
    t_origin : float
        Posunutí počátku času
    mask_func : callable(x, z)
        Funkce masky. Volá se pro všechny body mřížky, vrátí-li se True,
        není pole pro bod počítáno, vrácené pole v těchto souřadnicích
        obsahují NaN

    Returns
    -------
    E_x : 2D numpy.ndarray
        Xová složka E
    E_z : 2D numpy.ndarray
        Zová složka E
    """

    t_shifted = t - t_origin

    nrows = len(zs)
    ncols = len(xs)

    field_x = np.ndarray(shape=(nrows, ncols), dtype=np.float64)
    field_z = np.ndarray(shape=(nrows, ncols), dtype=np.float64)

    for row, z_orig in enumerate(zs):
        for col, x_orig in enumerate(xs):
            x = x_orig - x_origin
            z = z_orig - z_origin
            if mask_func:
                if mask_func(x, z):
                    field_x[row, col] = np.nan
                    field_z[row, col] = np.nan
                    continue

            e_x, e_z = E_at_point(x, z, t_shifted, q_x_func, q_z_func, C, EPS_0, Q)
            field_x[row, col] = e_x
            field_z[row, col] = e_z

    return field_x, field_z

@cython.embedsignature(True)
def E_theta_on_grid(xs, zs, double t,
                    q_x_func=None, q_z_func=None,
                    double C=0,
                    double EPS_0=0,
                    double Q=0,
                    double x_origin=0,
                    double z_origin=0,
                    double t_origin=0,
                    mask_func=None):
    """
    Počítá theta komponentu pole pohybujícího se náboje na mřížce bodů
    Parameters
    ----------
    xs : sorted iterable of floats
        Pole hodnot X mřížky
    zs : sorted iterable of floats
        Pole hodnot Y mřížky
    t : float
        Čas
    q_x_func : callable
        Funkce souřadnice X bodového náboje v čase
    q_z_func : callable
        Funkce souřadnice Z bodového náboje v čase
    C : float
        Rychlost světla
    EPS_0 : float
        Permitivita vakua
    Q : float
        Velikost náboje
    x_origin : float
        Posunutí počátku souřadnice X (aby bylo možné použít stejnou funkci
        polohy náboje na čase pro náboje v různých místech)
    z_origin : float
        Posunutí počátku souřadnice Z
    t_origin : float
        Posunutí počátku času
    mask_func : callable(x, z)
        Funkce masky. Volá se pro všechny body mřížky, vrátí-li se True,
        není pole pro bod počítáno, vrácené pole v těchto souřadnicích
        obsahují NaN

    Returns
    -------
    E_theta : 2D numpy.ndarray
        Theta složka E
    """

    cdef double t_shifted = t - t_origin

    nrows = len(zs)
    ncols = len(xs)

    field_theta = np.ndarray(shape=(nrows, ncols), dtype=np.float64)

    for row in range(nrows):
        for col in range(ncols):
            x = xs[col] - x_origin
            z = zs[row] - z_origin
            if mask_func:
                if mask_func(x, z):
                    field_theta[row, col] = np.nan
                    continue

            E_x, E_z = E_at_point(x, z, t_shifted, q_x_func, q_z_func, C, EPS_0, Q)
            theta = np.arctan2(abs(x), z)
            e_theta_x = np.cos(theta)
            e_theta_z = -np.sin(theta)

            field_theta[row, col] = E_x * e_theta_x * np.sign(x) + E_z * e_theta_z

    return field_theta
