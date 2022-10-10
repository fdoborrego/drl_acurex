import logging
import numpy as np
import random as rn
import scipy.optimize

logger = logging.getLogger(__name__)
logger.disabled = True

logger.setLevel(logging.DEBUG)  # DEBUG, INFO, WARNING, ERROR, EXCEPTION, CRITICAL
log_format = '[%(levelname)s - %(name)s] (%(asctime)s) %(message)s'
formatter = logging.Formatter(log_format)

consoleh = logging.StreamHandler()
consoleh.setFormatter(formatter)
logger.addHandler(consoleh)

fileh = logging.FileHandler('../log/logs.log')
fileh.setFormatter(formatter)
logger.addHandler(fileh)


class MPC:
    """
    Model Predictive Controller (MPC).
    """

    def __init__(self, env, prediction_horizon=5, control_horizon=3):

        # Parámetros del control predictivo
        self.prediction_horizon = prediction_horizon
        self.control_horizon = control_horizon

        self.x0 = np.tile(np.array([(env.low_action + env.high_action) / 2]), self.control_horizon)
        self.x_bounds = scipy.optimize.Bounds(np.tile(env.low_action, self.control_horizon),
                                              np.tile(env.high_action, self.control_horizon))

        # Función de coste
        self.cost_function = self._net_power

    def choose_action(self, environment):
        """
        Predicción.

        Cálculo de la acción de control óptima dado un estado del escenario.
        """
        # Control predictivo
        action = scipy.optimize.minimize(self.cost_function,
                                         self.x0,
                                         args=(environment, self.prediction_horizon, self.control_horizon),
                                         bounds=self.x_bounds,
                                         method='TNC', options={'gtol': 1e-6, 'disp': False})

        return action.x[0]

    @staticmethod
    def _net_power(x, *args):
        """
        Función de coste.

        Cálculo de la función de coste a minimizar por el Model Predictive Controller (MPC).

                J = -W + psi * (max(Tout-Tout_max, Tout_min - Tout, 0)/Tout_max)**2 + eps * (q(k) - q(k-1))**2
        """

        # Parámetros de función de coste
        psi, eps = 1e7, 1e5

        # Datos de entrada
        env, pred_horizon, cont_horizon = args

        # Construcción de acción de control
        q = np.concatenate([x, np.tile(x[-1], pred_horizon - cont_horizon)])

        # Inicialización
        in_temp = env.history['inlet_temperature'][-1] if env.history['inlet_temperature'] \
            else (env.ACUREXPlant.min_inlet_temp + env.ACUREXPlant.max_inlet_temp) / 2

        amb_temp = env.history['ambient_temperature'][-1] if env.history['ambient_temperature'] \
            else 25 + 2 * rn.random()

        current_output = env.state[0]

        # Predicción de T
        cost = 0
        for k in range(pred_horizon):

            for i in range(1, env.nsim_per_control + 1):
                step = env.step_count + (k * env.nsim_per_control + i)

                # Datos de entrada
                irr = env.irradiance_profile[step]
                in_temp = (600 - env.sim_time) / 600 * in_temp + env.sim_time / 600 * (current_output - 90)
                no = env.ACUREXPlant.calculate_geometric_efficiency(env.step_to_time(step), env.current_date)

                # Predicción
                current_output = env.ACUREXPlant.predict(current_output, q[k], irr, in_temp, amb_temp, no)

                # Datos de salida
                flow = q[k]
                last_flow = q[k - 1] if i > 0 else env.state[(1 + env.n_past_outputs)]

                outlet_temp = current_output
                outlet_power = env.ACUREXPlant.calculate_thermal_power(current_output, flow/1000)

                inlet_temp = in_temp
                inlet_power = env.ACUREXPlant.calculate_thermal_power(inlet_temp, flow/1000)

                # Función de coste
                power = outlet_power - inlet_power
                output_penalty = (max(outlet_temp - env.ACUREXPlant.max_outlet_opt_temp,
                                      env.ACUREXPlant.min_outlet_opt_temp - outlet_temp,
                                      0) / env.ACUREXPlant.max_outlet_opt_temp) ** 2
                action_penalty = (flow - last_flow) ** 2

                cost += -(power - psi * output_penalty - eps * action_penalty) * 1e-5

                logger.debug('[' + str(step) + '] ' +
                             'Net Power: ' + str(power) +
                             '; Output Penalty: ' + str(psi * output_penalty) +
                             '; Action Penalty: ' + str(eps * action_penalty))

        return cost
