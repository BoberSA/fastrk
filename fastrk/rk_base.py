"""
Algorithms for ode propagation with adaptive step selection
using explicit embedded runge-kutta method (rk_step):
- select_initial_step: select size of first step
- rk_variable_step: make an adaptive rk step according to tolerance
- rk_prop: integrate ode from s0, t0 to t (Cauchy's problem)
- event_detector: root separation for given event function
- prop_event: calculate event function from s0, t0 to t
- calc_root_brentq: root calculation procedure for events time location
- accurate_events: calculate state and time of given events
- rk_prop_ev: integrate ode from s0, t0 up to terminal events or time t (boundary problem)

Parts of code was used (and rewrited) from scipy.integrate.solve_ivp:
https://github.com/scipy/scipy/tree/v1.7.0/scipy/integrate/_ivp
and scipy.optimize.brentq:
https://github.com/scipy/scipy/blob/v1.7.0/scipy/optimize/zeros.py
"""

import numpy as np

# minimum feasible tolerance
EPS = np.finfo(float).eps

# rk_variable_step constants
SAFETY = 0.9  # Multiply steps computed from asymptotic behaviour of errors by this
MIN_FACTOR = 0.2  # Minimum allowed decrease in a step size
MAX_FACTOR = 10  # Maximum allowed increase in a step size

# rk_prop constants
N_STEPS = 2 ** 13  # 8192, standard array size for rk steps
N_STEPS_MUL = 2  # Array size multiplier when array is filled up
N_STEPS_MAX = 2 ** 24  # 16777216, maximum allowed steps count

# rk_prop_ev constants
N_EVENTS = 2 ** 10  # 1024, standard events count
N_EVENTS_MUL = 2  # Array size multiplier when array is filled up
N_EVENTS_MAX = 2 ** 24  # 16777216, maximum allowed events count


def rms_norm(x):
    return (x @ x / x.size) ** 0.5


def _resize_2darray_axis0(arr, new_size):
    new_arr = np.empty((new_size, arr.shape[1]), dtype=arr.dtype)
    new_arr[:min(arr.shape[0], new_size)] = arr
    return new_arr


def rk_variable_step(fode, t, s, h_abs, direction, t_bound, max_step, atol, rtol, out, *fargs):
    """
    Make one RK step of h_abs size in given direction and calculate size of the next step.

    :param fode: right part of ODE system, should be @cfunc
    :param s: current state
    :param t: current time
    :param h_abs: absolute value of current step
    :param direction: step direction (+1/-1)
    :param t_bound: boundary time
    :param max_step: maximum allowed step size
    :param atol: absolute tolerance
    :param rtol: relative tolerance
    :param out: (writable) array for time and state after successful step
    :param fargs: fode additional arguments
    :return: assumption for size of next step
    """
    min_step = 10 * np.abs(np.nextafter(t, direction * np.inf) - t)
    error_exponent = -1 / (RK_ORDER[1] + 1)

    if h_abs > max_step:
        h_abs = max_step
    elif h_abs < min_step:
        h_abs = min_step

    step_accepted = False
    step_rejected = False

    while not step_accepted:
        if h_abs < min_step:
            raise ValueError('Step size becomes too small')

        h = h_abs * direction
        t_new = t + h

        if direction * (t_new - t_bound) > 0:  # in case of last step
            t_new = t_bound

        h = t_new - t
        h_abs = abs(h)

        sarr = rk_step(fode, t, s, h, *fargs)
        s_new = sarr[:, 0]
        s_err = sarr[:, 1]

        scale = atol + np.maximum(np.abs(s), np.abs(s_new)) * rtol
        error_norm = rms_norm(s_err / scale)  # _estimate_error_norm(self.K, h, scale)

        if error_norm < 1:
            if error_norm == 0:
                factor = MAX_FACTOR
            else:
                factor = min(MAX_FACTOR, SAFETY * error_norm ** error_exponent)

            if step_rejected:
                factor = min(1, factor)

            h_abs *= factor

            step_accepted = True
        else:
            h_abs *= max(MIN_FACTOR, SAFETY * error_norm ** error_exponent)
            step_rejected = True

    out[0] = t_new
    out[1:] = s_new

    return h_abs


def select_initial_step(fode, t0, s0, direction, rtol, atol, *fargs):
    """
    Select good initial step size.
    See scipy.integrate.solve_ivp for details.

    :param fode: right part of ODE system, should be @cfunc
    :param t0: initial time
    :param s0: initial state
    :param direction: step direction (+1/-1)
    :param atol: absolute tolerance
    :param rtol: relative tolerance
    :param fargs: fode additional arguments
    :return:
    """
    if s0.size == 0:
        return np.inf

    f0 = fode(t0, s0, *fargs)
    scale = atol + np.abs(s0) * rtol
    d0 = rms_norm(s0 / scale)
    d1 = rms_norm(f0 / scale)
    if d0 < 1e-5 or d1 < 1e-5:
        h0 = 1e-6
    else:
        h0 = 0.01 * d0 / d1

    s1 = s0 + h0 * direction * f0
    f1 = fode(t0 + h0 * direction, s1, *fargs)
    d2 = rms_norm((f1 - f0) / scale) / h0

    if d1 <= 1e-15 and d2 <= 1e-15:
        h1 = max(1e-6, h0 * 1e-3)
    else:
        h1 = (0.01 / max(d1, d2)) ** (1 / (RK_ORDER[1] + 1))

    return min(100 * h0, h1)


def rk_prop(fode, s0, t0, t, max_step, rtol, atol, *fargs):
    """
    Integrate ODE from t0, s0 to t.
    See scipy.integrate.solve_ivp for details.

    :param fode: right part of ODE system, should be @cfunc
    :param s0: initial state
    :param t0: initial time
    :param t: boundary time
    :param max_step: maximum allowed step size
    :param rtol: relative tolerance
    :param atol: absolute tolerance
    :param fargs: fode additional arguments
    :return: array [[time, state]] for each integrator step
    """
    direction = 1. if t >= t0 else -1.
    h_abs = select_initial_step(fode, t0, s0, direction, rtol, atol, *fargs)
    # h_abs = abs(first_step)

    n_steps = N_STEPS
    trajectory = np.empty((n_steps, s0.size + 1), dtype=s0.dtype)
    trajectory[0, 0] = t0
    trajectory[0, 1:] = s0

    i = 0
    while abs(trajectory[i, 0] - t) > EPS:
        h_abs = rk_variable_step(fode, trajectory[i, 0], trajectory[i, 1:],
                                 h_abs, direction, t, max_step, rtol, atol,
                                 trajectory[i + 1], *fargs)
        i += 1
        if i >= n_steps - 1:
            # resize array
            n_steps *= N_STEPS_MUL
            if n_steps > N_STEPS_MAX:
                raise RuntimeError('Maximum allowed steps count exceeded')
            tmp = trajectory
            trajectory = _resize_2darray_axis0(trajectory, n_steps)

    return trajectory[:i + 1]


def event_detector(call_event, t, s, evvals, it, counters, values, terminals, directions, counts, evout, n_evout):
    """
    Separate roots of event functions, i.e. time moments when events triggered.

    :param call_event: call_event function
    :param t: current time
    :param s: current state
    :param evvals: array with event function values at previous times
    :param it: number of current iteration (index of state in trajectory)
    :param counters: current event counters
    :param values: array of values for event functions
    :param terminals: array of bools (whether event is terminal)
    :param directions: array of direction for events
    :param counts: array of
    :param evout: (writable) array [[event_index, event_counter, trajectory_index]]
    :param n_evout: (writable) array of one int (size of evout)
    :return:
    """
    terminal = False
    n_events = evvals.shape[1]
    for i in range(n_events):
        evvals[it, i] = call_event(t, s, values, i)

    if it == 0:
        counters[...] = counts
        return 0

    for i in range(n_events):
        cur_val = evvals[it][i]
        prev_val = evvals[it - 1][i]

        f1 = (prev_val < 0) and (cur_val > 0) and ((directions[i] == 1) or (directions[i] == 0))
        f2 = (prev_val > 0) and (cur_val < 0) and ((directions[i] == -1) or (directions[i] == 0))
        if (f1 or f2) and ((counters[i] == -1) or (counters[i] > 0)):
            if counters[i] > 0:
                counters[i] -= 1

            cnt = -1 if counters[i] == -1 else counts[i] - counters[i]
            # event index, event trigger counter, index of state before event
            evout[n_evout[0]] = np.array([i, cnt, it - 1])
            n_evout[0] += 1

            if terminals[i] and ((counters[i] == -1) or (counters[i] == 0)):
                terminal = True

    if terminal:
        return -1

    return 0


def prop_event(call_event, values, idx,
               _fode, _s0, _t0, _t, _max_step, _rtol, _atol, *fargs):
    """
    Propagate ODE state up to t and calculate value of [idx] event function
    :param call_event: call_event function
    :param values: events values
    :param idx: event index
    :param _rest: rk_prop arguments
    :param fargs: _fode additional arguments
    :return:
    """
    trj = rk_prop(_fode, _s0, _t0, _t, _max_step, _rtol, _atol, *fargs)
    return call_event(trj[-1, 0], trj[-1, 1:], values, idx)


def calc_root_brentq(xa, xb, xtol, rtol, maxiter,
                     _fode, _s0, _max_step, _rtol, _atol,
                     __call_event, __values, __idx, *fargs):
    """
    Brent's root finding algorithm for prop_event(t) function.
    See scipy.optimize.brentq for details.

    :param xa: left side of segment
    :param xb: right side of segment
    :param xtol: absolute tolerance by function argument
    :param rtol: relative tolerance
    :param maxiter: maximum number of iterations
    :param rest: prop_event arguments
    :return: <x>, where f(<x>) = 0 with specified tolerance
    """

    xpre, xcur = xa, xb
    xblk, fblk = 0., 0.
    spre, scur = 0., 0.

    fpre = prop_event(__call_event, __values, __idx, _fode, _s0, xa, xpre, _max_step, _rtol, _atol, *fargs)
    fcur = prop_event(__call_event, __values, __idx, _fode, _s0, xa, xcur, _max_step, _rtol, _atol, *fargs)
    if fpre * fcur > 0:
        raise ValueError('The event function has the same signs at both ends of the segment')

    if fpre == 0:
        return xpre

    if fcur == 0:
        return xcur

    for i in range(maxiter):
        if fpre * fcur < 0:
            xblk = xpre
            fblk = fpre
            spre = scur = xcur - xpre

        if abs(fblk) < abs(fcur):
            xpre = xcur
            xcur = xblk
            xblk = xpre

            fpre = fcur
            fcur = fblk
            fblk = fpre

        delta = (xtol + rtol * abs(xcur)) / 2
        sbis = (xblk - xcur) / 2
        if fcur == 0 or abs(sbis) < delta:
            return xcur

        if abs(spre) > delta and abs(fcur) < abs(fpre):
            if xpre == xblk:
                stry = -fcur * (xcur - xpre) / (fcur - fpre)
            else:
                dpre = (fpre - fcur) / (xpre - xcur)
                dblk = (fblk - fcur) / (xblk - xcur)
                stry = -fcur * (fblk * dblk - fpre * dpre) / (dblk * dpre * (fblk - fpre))

            if 2 * abs(stry) < min(abs(spre), 3 * abs(sbis) - delta):
                spre = scur
                scur = stry
            else:
                spre = sbis
                scur = sbis
        else:
            spre = sbis
            scur = sbis

        xpre = xcur
        fpre = fcur
        if abs(scur) > delta:
            xcur += scur
        else:
            xcur += delta if sbis > 0 else -delta

        fcur = prop_event(__call_event, __values, __idx, _fode, _s0, xa, xcur, _max_step, _rtol, _atol, *fargs)

    raise ValueError('Convergence error')


def accurate_events(trajectory, evout, accurates,
                    _call_event, _values, _xtol, _rtol, _maxiter,
                    __fode, __max_step, __rtol, __atol, *fargs):
    """
    Calculate event time and state where it needs to with given tolerance.

    :param trajectory: calculated trajectory (array of time-states)
    :param evout: array of [event_index, event_counter, trajectory_index]
    :param accurates: array of bools (True if event should be calculated accurate)
    :param _rest: calc_root_brentq, rk_prop arguments
    :param fargs: _fode additional arguments
    :return: event-states-array [[event_index, event_counter, time, state]]
    """
    # evout [[event_index, event_counter, trajectory_index]]
    n = evout.shape[0]
    res = np.empty((n, trajectory.shape[1] + 2), dtype=trajectory.dtype)

    for i in range(n):
        ei = evout[i, 0]
        ti = evout[i, 2]
        t0 = trajectory[ti, 0]
        t1 = trajectory[ti + 1, 0]
        s0 = trajectory[ti, 1:]
        res[i, :2] = evout[i, :2]
        if accurates[ei]:
            t = calc_root_brentq(t0, t1, _xtol, _rtol, _maxiter,
                                 __fode, s0, __max_step, __rtol, __atol,
                                 _call_event, _values, ei, *fargs)
            res[i, 2:] = rk_prop(__fode, s0, t0, t, __max_step, __rtol, __atol, *fargs)[-1]
        else:
            res[i, 2:] = trajectory[ti + 1]

    return res


def rk_prop_ev(fode, s0, t0, t, max_step, rtol, atol,
               _values, _terminals, _directions, _counts, _accurates,
               __call_event, __xtol, __rtol, __maxiter, *fargs):
    """
    Integrate ODE from t0, s0 up to any terminal event or time t.

    :param fode: right part of ODE system, should be @cfunc
    :param s0: initial state
    :param t0: initial time
    :param t: end time
    :param max_step: maximum step size
    :param rtol: relative tolerance
    :param atol: absolute tolerance
    :param _rest: event_detector, rk_variable_step arguments
    :param fargs: fode additional arguments
    :return: trajectory-array, event-states-array
    """
    direction = 1. if t >= t0 else -1.
    h_abs = select_initial_step(fode, t0, s0, direction, rtol, atol, *fargs)

    n_steps = N_STEPS
    trajectory = np.empty((n_steps, s0.size + 1), dtype=s0.dtype)
    trajectory[0, 0] = t0
    trajectory[0, 1:] = s0

    n_events = N_EVENTS
    ev_n = _terminals.size
    evvals = np.empty((n_steps, ev_n), dtype=s0.dtype)
    counters = np.empty(ev_n, dtype=np.int32)
    evout = np.empty((n_events, 3), dtype=np.int32)
    n_evout = np.zeros(1, dtype=np.int32)

    # first call
    event_detector(__call_event, t, s0, evvals, 0, counters,
                   _values, _terminals, _directions, _counts,
                   evout, n_evout)
    i = 0
    while abs(trajectory[i, 0] - t) > EPS:
        h_abs = rk_variable_step(fode, trajectory[i, 0], trajectory[i, 1:],
                                 h_abs, direction, t, max_step, rtol, atol,
                                 trajectory[i + 1], *fargs)
        i += 1

        if i >= n_steps - 1:
            n_steps *= N_STEPS_MUL
            if n_steps > N_STEPS_MAX:
                raise RuntimeError('Maximum allowed steps count exceeded')
            tmp = trajectory
            trajectory = _resize_2darray_axis0(trajectory, n_steps)
            evvals = _resize_2darray_axis0(evvals, n_steps)

        trm = event_detector(__call_event, trajectory[i, 0], trajectory[i, 1:], evvals, i,
                             counters, _values, _terminals, _directions, _counts,
                             evout, n_evout)

        if n_evout[0] >= n_events - 1:
            n_events *= N_EVENTS_MUL
            if n_steps > N_EVENTS_MAX:
                raise RuntimeError('Maximum allowed count of event records exceeded')
            tmp = evout
            evout = _resize_2darray_axis0(evout, n_events)

        if trm != 0:
            break

    event_out = accurate_events(trajectory, evout[:n_evout[0]], _accurates,
                                __call_event, _values, __xtol, __rtol, __maxiter,
                                fode, max_step, rtol, atol, *fargs)

    return trajectory[:i + 1], event_out
