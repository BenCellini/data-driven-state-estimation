
import numpy as np
import pandas as pd
import sympy as sp
import matplotlib.pyplot as plt
import itertools
import figurefirst as fifi
import figure_functions as ff

from pybounds import SlidingEmpiricalObservabilityMatrix, SlidingFisherObservability, colorline
from drone_model import DroneSimulator


class DroneObservability:
    """ Run empirical observability analysis on a fly trajectory.
    """

    def __init__(self, simulator=None, dt=0.1, states=None, sensors=None, time_steps=None,
                 mpc_horizon=10, mpc_control_penalty=1e-2, control_mode='velocity_body_level', polar=False):
        """ Run.
        """

        self.polar = polar

        if self.polar:
            self.z_function = z_function
        else:
            self.z_function = None

        # Set the states, sensors, & time-steps to use
        if states is None:
            self.states = [['x', 'y', 'z',
                            'v_x', 'v_y', 'v_z',
                            'phi', 'theta', 'psi',
                            'omega_x', 'omega_y', 'omega_z',
                            'w', 'zeta',
                            'm', 'I_x', 'I_y', 'I_z', 'C']]
        else:
            self.states = states.copy()

        if sensors is None:
            # self.sensors = [['psi', 'gamma', 'beta', 'r']]
            self.sensors = [['psi', 'gamma'],
                            ['psi', 'gamma', 'beta'],
                            ['psi', 'gamma', 'beta', 'phi', 'theta'],
                            ['psi', 'gamma', 'beta', 'r'],
                            ['psi', 'gamma', 'beta', 'r', 'phi', 'theta'],
                            ['psi', 'gamma', 'beta', 'a'],
                            ['psi', 'gamma', 'beta', 'g'],
                            ['psi', 'gamma', 'beta', 'r', 'a'],
                            ['psi', 'gamma', 'beta', 'a', 'g'],
                            ['psi', 'gamma', 'beta', 'a', 'r', 'phi', 'theta'],
                            ['psi', 'gamma', 'beta', 'a', 'g', 'phi', 'theta']
                            ]
        else:
            self.sensors = sensors.copy()

        if time_steps is None:
            self.time_steps = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        else:
            self.time_steps = time_steps.copy()

        # Unique sensors
        sensors_all = []
        for s in self.sensors:
            sensors_all = sensors_all + s

        self.sensors_all = list(set(sensors_all))

        # Create simulator
        if simulator is None:
            self.simulator = DroneSimulator(dt=dt, mpc_horizon=mpc_horizon, r_u=mpc_control_penalty,
                                            control_mode=control_mode)
        else:
            self.simulator = simulator(dt=dt, mpc_horizon=mpc_horizon, r_u=mpc_control_penalty,
                                       control_mode=control_mode)

        # State names in polar coordinates
        if self.polar:
            replacement_polar_states = {'v_x': 'g', 'v_y': 'beta'}
            self.states_polar = []
            for state_list in self.states:
                self.states_polar.append([replacement_polar_states.get(x, x) for x in state_list])

            self.states = self.states_polar
            self.states_polar_all = [replacement_polar_states.get(x, x) for x in self.simulator.state_names.copy()]
        else:
            self.states_polar = [None] * len(self.states)
            self.states_polar_all = None

        # Every combination of states, sensors, & time-steps
        self.comb = list(itertools.product(self.states, self.sensors, self.time_steps))
        self.n_comb = len(self.comb)

        # Set Observability & Fisher Information parameters
        self.SEOM = []
        self.data_dict = {}

    def run(self, v_x=None, v_y=None, psi=None, z=None, w=None, zeta=None,
            R=0.1, lam=1e-6, align_to_initial_time=False):
        """ Run.
        """

        # Update the setpoints
        self.simulator.update_setpoint(v_x=v_x, v_y=v_y, psi=psi, z=z, w=w, zeta=zeta)

        # Reconstruct trajectory with MPC
        t_sim, x_sim, u_sim, y_sim = self.simulator.simulate(x0=None, mpc=True, return_full_output=True)

        # Get simulation data
        sim_data = pd.DataFrame(y_sim)
        sim_data.insert(loc=0, column='time', value=t_sim)
        # sim_data['time'] = sim_data['time'].values - sim_data['time'].values[0]

        # Construct observability matrix in sliding windows
        time_window_max = np.max(np.array(self.time_steps))

        self.SEOM = SlidingEmpiricalObservabilityMatrix(self.simulator, t_sim, x_sim, u_sim,
                                                        w=time_window_max, eps=1e-4,
                                                        z_function=self.z_function,
                                                        z_state_names=self.states_polar_all)

        # Dictionary to store data for trajectory
        data_dict = {'states': [], 'sensors': [], 'time_steps': [], 'sim_data': [], 'error_variance': [],
                     'O_sliding': self.SEOM.O_df_sliding}

        for n, c in enumerate(self.comb):  # each state, sensor, time-step combination
            o_states = c[0]
            o_sensors = c[1]
            time_window = c[2]
            o_time_steps = np.arange(0, time_window)

            # Run Fisher Information observability for each sliding window
            # O_sliding = self.SEOM.get_observability_matrix()
            SFO = SlidingFisherObservability(self.SEOM.O_df_sliding, time=self.SEOM.t_sim, lam=lam, R=R,
                                             states=o_states, sensors=o_sensors,
                                             time_steps=o_time_steps, w=time_window)

            EV_aligned = SFO.get_minimum_error_variance()

            # Align & rename observability data
            EV_aligned_no_nan = EV_aligned.copy()
            # EV_aligned_no_nan = EV_aligned_no_nan.fillna(method='bfill').fillna(method='ffill')

            # Align to initial time index
            if align_to_initial_time:
                time_initial_index = np.where(EV_aligned_no_nan['time_initial'].values <= 0)[0][-1]
                EV_aligned_no_nan = EV_aligned_no_nan.shift(-time_initial_index, axis=0)
                EV_aligned_no_nan = EV_aligned_no_nan.dropna(axis=0)
                time_end_index = np.where(EV_aligned_no_nan['time_initial'].values
                                          >= EV_aligned_no_nan['time_initial'].values[-1])[0][0]
                EV_aligned_no_nan = EV_aligned_no_nan.iloc[0:time_end_index + 1, :]

            EV_aligned_rename = EV_aligned_no_nan.copy()
            EV_aligned_rename.columns = ['o_' + item for item in EV_aligned_no_nan.columns]

            # Combine simulation & observability data
            sim_data_new = sim_data.copy()
            sim_data_all = pd.concat([sim_data_new, EV_aligned_rename], axis=1)

            # Append data to dictionary
            data_dict['states'].append(o_states)
            data_dict['sensors'].append(o_sensors)
            data_dict['time_steps'].append(time_window)
            data_dict['error_variance'].append(EV_aligned_no_nan)
            data_dict['sim_data'].append(sim_data_all)

        self.data_dict = data_dict

        return data_dict

    def plot_trajectory(self, start_index=0, dpi=200, size_radius=0.1):
        """ Plot the trajectory.
        """

        sim_data = self.data_dict['sim_data'][0]

        fig, ax = plt.subplots(1, 1, figsize=(3 * 1, 3 * 1), dpi=dpi)

        x = sim_data['x'].values[start_index:]
        y = sim_data['y'].values[start_index:]
        heading = sim_data['psi'].values[start_index:]
        time = sim_data['time'].values[start_index:]

        ff.plot_trajectory(x, y, heading,
                           color=time,
                           ax=ax,
                           size_radius=size_radius,
                           nskip=0)

        fifi.mpl_functions.adjust_spines(ax, [])


# Coordinate transformation function
def z_function(X):
    # Old states as sympy variables
    x, y, z, v_x, v_y, v_z, phi, theta, psi, omega_x, omega_y, omega_z, w, zeta, m, I_x, I_y, I_z, C = X

    # Expressions for new states in terms of old states
    g = (v_x ** 2 + v_y ** 2) ** (1 / 2)  # ground speed magnitude
    beta = sp.atan(v_y / v_x)  # ground speed angle

    # Define new state vector
    z = [x, y, z, g, beta, v_z, phi, theta, psi, omega_x, omega_y, omega_z, w, zeta, m, I_x, I_y, I_z, C]
    return sp.Matrix(z)
