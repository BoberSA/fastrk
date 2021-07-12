# FastRK

- developed as fast alternative for subset of [scipy.integrate.ode](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.ode.html) methods (i.e. `DOP853`);
- is a python code generator for Ordinary Differential Equations (ODE) propagation;
- uses explicit embedded Runge-Kutta (ERK) methods adaptive step technique;
- calculates events using event functions (like [scipy.integrate.solve_ivp](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html)); 
- is jit-compiled by `Numba`;
  * compiled code cached on SSD/HDD to prevent unnecessary recompilation;
- OS-independent (to the same extent as `Numba`);
- contains Butcher Tables for several ERK methods:
    - Dormand and Prince 6(5)8M;
    - Dormand and Prince 8(7)13M;
      * ~2-3x faster than `DOP853` from `scipy.integrate.ode`
    - Verner's 8(9)16;
- user-defined Butcher Tables also supported; 
- generated code is open and user-modifiable; 
- 

Butcher Tables was adapted from [TrackerComponentLibrary](https://github.com/USNavalResearchLaboratory/TrackerComponentLibrary).

## Installation

    pip install fastrk

## Fast example

```python:examples/fast_example.py```


## Detailed examples

[Example 0: Propagate spacecraft motion in Circular Restricted Three Body Problem](https://github.com/BoberSA/fastrk/blob/master/examples/ex0_propagate_crtbp.ipynb)

[Example 1: Calculate events in Circular Restricted Three Body Problem](https://github.com/BoberSA/fastrk/blob/master/examples/ex1_calculate_events.ipynb)

## Core Developer
Stanislav Bober, [MIEM NRU HSE](https://miem.hse.ru/), [IKI RAS](http://iki.rssi.ru/)
