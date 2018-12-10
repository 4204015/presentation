import sympy as sp
import math
import symbtools.modeltools as mt
import matplotlib.pyplot as plt

from visualisation import PendelWagenSystem as PWSVis
from visualisation import MehrMassenSystem as MMSVis

import numpy as np
import scipy.integrate as sci
import symbtools as st

from sklearn.base import BaseEstimator
from bokeh.plotting import show
from bokeh.layouts import row, gridplot
from bokeh.io import push_notebook
from bokeh.models import Band
from bokeh.models import Range1d


class Simulator(BaseEstimator):
    def __init__(self, model, x0, t0, T=None, input_func_gen=None, func_parameters=None,
                 step_size=0.1e-3, solver='odeint', state_idx=None):

        self.model = model
        self.input_func_gen = input_func_gen
        self.func_parameters = func_parameters
        self.x0 = x0
        self.t0 = t0
        self.T = T
        self.step_size = step_size
        self.solver = solver
        self.state_idx = state_idx

    def solve(self, param, T=None, additional_time=0, use_sp2c=True,
              start_end=None, input_func=None, get_input=False, debug=False):
        assert bool(self.T) != bool(T)  # xor
        assert not any(np.isnan(param))

        T_spline = self.T if not T else T
        T_sim = T_spline + additional_time
        tt = np.arange(self.t0, self.t0 + T_sim, self.step_size)

        if not input_func:
            def input_func(t):
                return self.input_func_gen(t=t, t_range=[self.t0, self.t0 + T_spline], param=param,
                                           start_end=self.func_parameters.get(
                                               'start_end', (start_end if start_end else (0.0, 0.0))))

        sim_func = self.model.create_simfunction(input_function=input_func,
                                                 use_sp2c=use_sp2c)
        
        if debug:
            return sim_func
        
        if self.solver == 'odeint':
            res = sci.odeint(func=sim_func, y0=self.x0, t=tt, rtol=1e-6)
        else:

            res = None # TODO
        
        if get_input:
            return res, tt, [input_func(t) for t in tt]
        else:
            if self.state_idx is not None:
                return res[:, self.state_idx], tt
            else:
                return res, tt

    def plot_input_func(self, param, start_end=None):
        tt = np.arange(self.t0, self.t0 + self.T, self.step_size)

        def input_func(t):
            return self.input_func_gen(t=t, t_range=[self.t0, self.t0 + self.T], param=param,
                                       start_end=self.func_parameters.get('start_end', start_end))

        u = [input_func(t_) for t_ in tt]

        fig = plt.figure()
        plt.plot(tt, u)
        plt.xlabel("t")
        plt.ylabel("u(t)")
        plt.grid()
        plt.tight_layout()


class LocalOptimizer:
    def __init__(self, simulator):
        self.simulator = simulator

    def _esimate_jacobian(self, target, dx=0.001):
        Fa = np.array(self.simulator.solve(self.simulator.param)[-1, :]) - target
        J = np.zeros((len(self.simulator.x0), len(self.simulator.param)))

        for i in range(len(self.simulator.param)):
            param = self.simulator.param
            param[i] = param[i] + dx

            Fb = np.array(self.simulator.solve(param)[-1, :]) - target
            J[:, i] = np.gradient([Fa, Fb], dx, axis=0)[0]

        return J, Fa

    def search(self, target, search_algorithm='gd', stopping=1000,
               learning_rate=0.001, weights=None, alpha=0.9, beta1=0.9, beta2=0.999,
               epsilon=10e-8, plotter=None):

        assert len(target) == len(self.simulator.param)

        # inital guess for all parameters
        self.simulator.param = self.simulator.func_parameters.get(
            'initial')  # , np.hstack((np.random.uniform(*span) for span in self.func_parameters['range'])))

        params = np.array(self.simulator.param)
        objective_function = []

        if weights:
            W = np.diag(weights)
        else:
            W = np.identity(n=len(self.simulator.param))

        v = 0
        m = 0

        while len(objective_function) < stopping:
            t = len(objective_function) + 1
            J, F = self._esimate_jacobian(target)

            gradient = J.T @ W @ F

            objective = F.T @ W @ F
            objective_function.append(objective)

            if objective < 0.0001:# or objective > 2*objective_function[0]:
                msg = "Stopping because convergence" if objective < 0.0001 else "explosion of loss"
                print(msg + f": {objective}")
                plotter.update({'loss': objective, 'param': self.simulator.param, 'phase_space': F})
                break

            if search_algorithm == 'momentum':
                v = v * alpha + (1 - alpha) * gradient
                self.simulator.param -= learning_rate * v

            elif search_algorithm == 'adam':
                m = beta1 * m + (1 - beta1) * gradient
                v = beta2 * v + (1 - beta2) * (gradient) ** 2

                m_ = m / (1 - beta1 ** t)
                v_ = v / (1 - beta2 ** t)

                self.simulator.param -= learning_rate * m_ / (np.sqrt(v_) + epsilon)
            else:
                self.simulator.param -= learning_rate * gradient

            if plotter:
                plotter.update({'loss': objective, 'param': self.simulator.param, 'phase_space': F})

            params = np.vstack((params, self.simulator.param))

        return params, objective_function


class TrainingPlotter:
    def __init__(self, *args, types):
        self.figures = args
        self.types = types
        assert len(self.figures) == len(types)
        self.plots = [getattr(fig, plot_type)([0], [0]) for fig, plot_type in zip(args, types)]

        if len(self.figures) > 3:
            f = list(self.figures) + [None]*(6-len(self.figures))
            grid = gridplot(np.reshape(f, (2, 3)).tolist())
            self.plot_handle = show(grid, notebook_handle=True)

        else:
            self.plot_handle = show(row(*self.figures), notebook_handle=True)

        self.i = 0

    def update(self, data_dict):
        for plot, fig in zip(self.plots, self.figures):

            try:
                try:
                    key, idx = fig.title.text.split('-')
                    data = data_dict[key][int(idx)]
                except ValueError:
                    data = data_dict[fig.title.text]
            except KeyError:
                if self.i == 0:
                    pass
                else:
                    continue

            new_data = dict()
            if self.i == 0:

                if fig.title.text == 'samples':
                    fig.xaxis.axis_label = 'phi'
                    fig.yaxis.axis_label = 'phi_dot'
                    new_data['x'], new_data['y'], *_ = [data[0], data[1]]

                elif fig.title.text == 'model_ranges':

                    new_data = {'lower': [1.0], 'upper': [1.0], 'x': [self.i], 'y': [1.0]}
                    plot.data_source.data.update(new_data)

                    self.band = Band(base='x', lower='lower', upper='upper', source=plot.data_source, level='underlay',
                                     fill_alpha=1.0, line_width=1, line_color='black')
                    fig.add_layout(self.band)
                    fig.xaxis.axis_label = 'iteration'
                else:
                    fig.xaxis.axis_label = 'iteration'

                    new_data['x'] = [self.i]
                    new_data['y'] = [data]


            else:
                if fig.title.text == 'samples':
                    new_data['x'] = plot.data_source.data['x'] + data[0]
                    new_data['y'] = plot.data_source.data['y'] + data[1]
                elif fig.title.text == 'model_ranges':
                    new_data['x'] = plot.data_source.data['x'] + [self.i]
                    new_data['y'] = plot.data_source.data['y'] + [data['y']]
                    new_data['lower'] = plot.data_source.data['lower'] + [data['lower']]
                    new_data['upper'] = plot.data_source.data['upper'] + [data['upper']]
                    fig.y_range = Range1d(np.min(plot.data_source.data['lower']), 2)
                else:
                    new_data['x'] = plot.data_source.data['x'] + [self.i]
                    new_data['y'] = plot.data_source.data['y'] + [data]


            plot.data_source.data.update(new_data)
        self.i += 1

        push_notebook(handle=self.plot_handle)

    def reset(self):
        self.i = 0

    def draw(self):
        pass


class PendelWagenSystem:
    def __init__(self, inverted=True, calc_coll_part_lin=True):

        self.inverted = inverted
        self.calc_coll_part_lin = calc_coll_part_lin
        # -----------------------------------------
        # Pendel-Wagen System mit hängendem Pendel
        # -----------------------------------------

        pp = st.symb_vector(("varphi", ))
        qq = st.symb_vector(("q", ))

        ttheta = st.row_stack(pp, qq)
        st.make_global(ttheta)

        params = sp.symbols('m1, m2, l, g, q_r, t, T')
        st.make_global(params)

        ex = sp.Matrix([1,0])
        ey = sp.Matrix([0,1])

        # Koordinaten der Schwerpunkte und Gelenke
        S1 = ex * q  # Schwerpunkt Wagen
        G2 = S1      # Pendel-Gelenk

        # Schwerpunkt des Pendels (Pendel zeigt für kleine Winkel nach oben)

        if inverted:
            S2 = G2 + mt.Rz(varphi) * ey * l

        else:
            S2 = G2 + mt.Rz(varphi) * -ey * l

        # Zeitableitungen der Schwerpunktskoordinaten
        S1d, S2d  = st.col_split(st.time_deriv(st.col_stack(S1, S2), ttheta))

        # Energie
        E_rot = 0 # (Punktmassenmodell)
        E_trans = (m1*S1d.T*S1d  +  m2*S2d.T*S2d) / 2

        E = E_rot + E_trans[0]

        V = m2*g*S2[1]

        # Partiell linearisiertes Model
        mod = mt.generate_symbolic_model(E, V, ttheta, [0, sp.Symbol("u")])
        mod.calc_state_eq()
        mod.calc_coll_part_lin_state_eq()

        self.mod = mod

    def get_sim_model(self, model_parameters):
        if self.calc_coll_part_lin:
            sim_mod = st.SimulationModel(self.mod.ff, self.mod.gg, self.mod.xx, model_parameters)
        else:
            sim_mod = st.SimulationModel(self.mod.f, self.mod.g, self.mod.xx, model_parameters)

        sim_mod.convert_to_c()
        return sim_mod


    def get_animation_model(self, l, car_width=0.1, car_height=0.05, pendulum_size=0.05):
        rod_length = -l if not self.inverted else l
        return PWSVis(rod_length=rod_length, car_width=car_width, car_height=car_height, pendulum_size=pendulum_size)


class MehrMassenSchwinger:
    def __init__(self, n_p=1):
        self.n_p = n_p  # passive Koordinate
        
        n_q = 1
        q = sp.symbols("q")
        if n_p > 0:
            pp = st.symb_vector("p1:{0}".format(n_p + 1))
        else:
            pp = []

        theta = st.row_stack(q, pp)
        thetadot = st.time_deriv(theta, theta)
        
        mass_sym = sp.symbols(", ".join([f'm{n}' for n in range(n_p+n_q)]))
        damping_sym = sp.symbols(", ".join([f'd{n}' for n in range(n_p+n_q)]))
        stiffness_sym = sp.symbols(", ".join([f'c{n}' for n in range(n_p+n_q)]))
        ttau = sp.symbols(", ".join([f'tau{n}' for n in range(n_p+n_q)]))
        
        if not isinstance(mass_sym, tuple):
            mass_sym = [mass_sym]
        if not isinstance(damping_sym, tuple):
            damping_sym = [damping_sym]
        if not isinstance(stiffness_sym, tuple):
            stiffness_sym = [stiffness_sym]
        if not isinstance(ttau, tuple):
            ttau = [ttau]

        # Koordinaten der Massenschwerpunkte
        S = [q] + [p for p in pp]
                
        # Zeitableitungen der Schwerpunktskoordinaten
        Sdot = [st.time_deriv(s, theta) for s in S]

        # Energie
        # Kinetische Energie
        T = 0.5 * sum([m * sdot**2 for m, sdot in zip(mass_sym, Sdot)])

        # Potentielle Energie
        if n_p > 0:
            #V = 0.5 * sum([c * p**2 for p, c in zip(pp, stiffness_sym)])
            
            S = [0] + S
            V = 0.5 * sum([c * (S[i+1] - S[i])**2  for i, c in enumerate(stiffness_sym)])
        else:
            V = 0

        # Partiell linearisiertes Model
        mod = mt.generate_symbolic_model(T, V, theta, ttau)
        
        
        Sdot = [0] + Sdot
        F_diss = [d * (Sdot[i+1] - Sdot[i]) for i, d in enumerate(damping_sym)] + [0]
        
        tau_subs = [[tau, - F_diss[i] + F_diss[i+1]] for i, tau in enumerate(ttau)]
        tau_subs[0][1] += ttau[0]
        
        mod.eqns = sp.simplify(mod.eqns.subs(tau_subs))
        mod.calc_state_eq()

        self.mod = mod
        self.mod.g = self.mod.g[:, 0]

    def get_sim_model(self, model_parameters):
        self.model_parameters = model_parameters
        sim_mod = st.SimulationModel(self.mod.f, self.mod.g, self.mod.xx, model_parameters)
        sim_mod.convert_to_c()
        return sim_mod

    def get_animation_model(self, body_height=0.4, body_width=0.4, body_distance=1.0):
        return MMSVis(body_height=body_height, body_width=body_width, num_bodies=self.n_p+1, body_distance=body_distance)

    def get_energy(self, res):
        states = res[0]
        T = 0.5 * (states[:, self.n_p+1:]**2 @ np.array([self.model_parameters[f'm{i}'] for i in range(self.n_p+1)]))
        states = [0] + states
        V = 0.5 * (np.array([states[:, i+1] - states[:, i] for i in range(self.n_p+1)]).T**2 @ 
                   np.array([self.model_parameters[f'c{i}'] for i in range(self.n_p+1)]))
        return np.array([T, V]).T, res[1]
    
    def get_forces(self, res):
        states = res[0]
        states = [0] + states
        F_c = np.array([states[:, i+1] - states[:, i] for i in range(self.n_p+1)]).T**2 * np.array([self.model_parameters[f'c{i}'] for i in range(self.n_p+1)])
        F_d = np.array(states[:, self.n_p+1:]) * np.array([self.model_parameters[f'd{i}'] for i in range(self.n_p+1)])
        return np.array([F_c, F_d]).T, res[1]


def logistic(x, L=1.0, k=1.0, x0=0.0):
    return L / (1.0 + np.exp(-k * (x-x0)))


class TrajectoryProblem:
    def __init__(self, simulator, bounds, weights=None, penalty_config=None, time_as_param=False,
                 specify_end=False, unique_angle=False, target=None):

        self.sim = simulator
        self.W = np.diag(weights) if weights else np.identity(len(bounds))
        self.penalty_config = penalty_config
        self.bounds = bounds  # e.g. [(-5, 5), (-10, 10)]
        self.time_as_param = time_as_param
        self.specify_end = specify_end
        self.unique_angle = unique_angle
        self.target=target

        if self.time_as_param and self.specify_end:
            raise NotImplementedError

    def get_penalties(self, states):
        penalties = np.zeros(self.W.shape[0])

        if self.penalty_config is None:
            return penalties

        for i, pen_opt in enumerate(self.penalty_config):
            if len(pen_opt.keys()) == 0:
                continue
            # get base value
            base = pen_opt['base']
            if base == 'final':
                v = states.T[i, -1]
            elif base == 'max':
                v = np.max(states.T[i, :])
            elif base == 'min':
                v = np.min(states.T[i, :])
            else:
                raise NotImplementedError

            # penalty calculation
            soft = pen_opt.get('soft', True)  # bool!
            kind = pen_opt['kind']

            L = 100
            s = 0.001

            value = pen_opt['value']
            if kind == 'range':
                if not soft:
                    penalties[i] = L if (v < value[0]) and (v > value[1]) else 0
                else:
                    x1_0 = value[0] * 1.1
                    k1 = math.log((L / s) - 1) / (0.1 * value[0])
                    x2_0 = value[1] * 1.1
                    k2 = math.log((L / s) - 1) / (0.1 * value[1])

                    penalties[i] = logistic(x=v, L=L, k=k2*np.sign(value[1]), x0=x2_0) -\
                                   logistic(x=v, L=L, k=k1*np.sign(value[0]), x0=x1_0) + L
            elif kind == 'max':
                if not soft:
                    penalties[i] = L if (not soft) and (v > value) else 0
                else:
                    x0 = value * 1.1
                    k = math.log((L / s) - 1) / (0.1 * value)
                    penalties[i] = logistic(x=v, L=L, k=k, x0=x0)
            elif kind == 'min':
                if not soft:
                    penalties[i] = L if (not soft) and (v < value) else 0
                else:
                    x0 = value * 1.1
                    k = math.log((L / s) - 1) / (0.1 * value)
                    penalties[i] = logistic(x=v, L=L, k=-k, x0=x0)
            else:
                raise NotImplementedError

        return penalties

    def fitness(self, dv):
        DV = np.load("DV.npy")
        np.save("DV.npy", np.vstack((DV, dv)))
        
        l = len(self.bounds)
        if self.time_as_param:
            T = dv[-1]
            l -= 1
        else:
            T = None

        end = 0.0
        if self.specify_end:
            end = dv[-1]
            l -= 1

        states = self.sim.solve(param=dv[0:l], T=T, start_end=(0.0, end))[0]
        penalties = self.get_penalties(states)

        final_state = states.T[:, -1]
        if self.unique_angle:
            final_state[0] %= (2 * np.pi)

        # Root Mean Squared Error of the weighted states
        if self.target:
            rms = np.sqrt((final_state-self.target).T @ self.W @ (final_state-self.target))
        else:
            rms = np.sqrt(final_state.T @ self.W @ final_state)

        self.final_state = final_state

        return [rms + np.sum(penalties)]

    def get_bounds(self):
        return tuple(zip(*self.bounds))


if __name__ is "__main__":
    MehrMassenSchwinger(init_pos=[0.0])