import gym
import math
import pickle
import logging
import scipy.io
import numpy as np
import random as rn
import matplotlib.pyplot as plt
from models.ConcentratedModel import ACUREXModel

plt.switch_backend('MACOSX')

logger = logging.getLogger(__name__)
logger.disabled = False

logger.setLevel(logging.DEBUG)  # DEBUG, INFO, WARNING, ERROR, EXCEPTION, CRITICAL
log_format = '[%(levelname)s - %(name)s] (%(asctime)s) %(message)s'
formatter = logging.Formatter(log_format)

consoleh = logging.StreamHandler()
consoleh.setFormatter(formatter)
# logger.addHandler(consoleh)

fileh = logging.FileHandler('../log/logs.log')
fileh.setFormatter(formatter)
logger.addHandler(fileh)


class ACUREXEnv(gym.Env):
    """
    Environment de la planta termosolar ACUREX. Modelo de parámetros concentrados.
    · State: [Tout(t), ..., Tout(t-Dy),
              q(t-1), ..., q(t-Du),
              I(t+Ddf), ..., I(t), ..., I(t-Ddp),
              Tin(t+Ddf), ..., Tin(t), ..., Tin(t-Ddp),
              Tamb(t+Ddf), ..., Tamb(t), ..., Tamb(t-Ddp),
              no(t+Ddf), ..., no(t), ..., no(t-Ddp),
              Wout(t), ..., Wout(t-Dy),
              Win(t), ..., Win(t-Dy)]
    · Action: [q(t)]
    · Reward: rew = net_power - error_penalty - slew_rate_penalty
    · Done: Tsim, min_temp, max_temp
    """

    environment_name = "ACUREX Solar Field - Concentrated Parameters Model"

    def __init__(self):

        """ Parámetros de configuración """
        # Parámetros de simulación
        self.sim_time = 0.25                    # Tiempo de integración (\Delta T_sim) [s]

        # Parámetros del estado
        self.n_past_outputs = 20                # Número de salidas pasadas consideradas en el estado
        self.n_past_actions = 20                # Número de acciones pasadas consideradas en el estado
        self.n_past_disturbances = 20           # Número de perturbaciones pasadas consideradas en el estado
        self.n_future_disturbances = 10         # Número de perturbaciones futuras consideradas en el estado

        self.dt_disturbances = 2 * 60           # Intervalo entre lecturas de las perturbaciones [s]

        """ Modelo utilizado """
        self.ACUREXPlant = ACUREXModel(integration_time=self.sim_time)

        """ Action space """
        self.high_action = np.array([self.ACUREXPlant.max_flow], dtype=np.float32)
        self.low_action = np.array([self.ACUREXPlant.min_flow], dtype=np.float32)
        self.action_space = gym.spaces.Box(low=self.low_action, high=self.high_action, dtype=np.float32)

        """ Observation space """
        min_out_power, max_out_power = \
            self.ACUREXPlant.calculate_thermal_power(self.ACUREXPlant.min_outlet_temp, self.ACUREXPlant.min_flow), \
            self.ACUREXPlant.calculate_thermal_power(self.ACUREXPlant.max_outlet_temp, self.ACUREXPlant.max_flow)

        min_in_power, max_in_power = \
            self.ACUREXPlant.calculate_thermal_power(self.ACUREXPlant.min_inlet_temp, self.ACUREXPlant.min_flow), \
            self.ACUREXPlant.calculate_thermal_power(self.ACUREXPlant.max_inlet_temp, self.ACUREXPlant.max_flow)

        # Estado
        self.high_state = \
            [self.ACUREXPlant.max_outlet_temp] * (self.n_past_outputs + 1) +\
            [self.ACUREXPlant.max_flow] * (self.n_past_actions + 1) +\
            [self.ACUREXPlant.max_irradiance] * (self.n_future_disturbances + 1 + self.n_past_disturbances) +\
            [self.ACUREXPlant.max_inlet_temp] * (self.n_future_disturbances + 1 + self.n_past_disturbances) +\
            [self.ACUREXPlant.max_ambient_temp] * (self.n_future_disturbances + 1 + self.n_past_disturbances) +\
            [self.ACUREXPlant.max_geometric_efficiency] * (self.n_future_disturbances + 1 + self.n_past_disturbances) +\
            [max_out_power] * (1 + self.n_past_outputs) +\
            [max_in_power] * (1 + self.n_past_outputs)
        self.high_state = np.array(self.high_state, dtype=np.float32)

        self.low_state = \
            [self.ACUREXPlant.min_outlet_temp] * (self.n_past_outputs + 1) +\
            [self.ACUREXPlant.min_flow] * (self.n_past_actions + 1) +\
            [self.ACUREXPlant.min_irradiance] * (self.n_future_disturbances + 1 + self.n_past_disturbances) +\
            [self.ACUREXPlant.min_inlet_temp] * (self.n_future_disturbances + 1 + self.n_past_disturbances) +\
            [self.ACUREXPlant.min_ambient_temp] * (self.n_future_disturbances + 1 + self.n_past_disturbances) +\
            [self.ACUREXPlant.min_geometric_efficiency] * (self.n_future_disturbances + 1 + self.n_past_disturbances) +\
            [min_out_power] * (self.n_past_outputs + 1) +\
            [min_in_power] * (self.n_past_disturbances + 1)
        self.low_state = np.array(self.low_state, dtype=np.float32)

        self.observation_space = gym.spaces.Box(low=self.low_state, high=self.high_state, dtype=np.float32)

        """ Parámetros útiles """
        # Número de iteraciones de simulación por periodo de muestreo
        self.nsim_per_sample = int(math.ceil(self.ACUREXPlant.sample_time / self.sim_time))

        # Número de iteraciones de simulación por periodo de control
        self.nsim_per_control = int(math.ceil(self.ACUREXPlant.control_time / self.sim_time))

        """ Inicialización del agente """
        # Configuración de la simulación
        self.config_params = {'irradiance_file': 0, 'start': 0, 'lapse': 0}

        self.start_time, self.stop_time = 9, 18
        self.current_date = {'day': 16, 'month': 11, 'year': 1998}
        self.irradiance_profile = []
        self.Tsim = 1

        self.load_file(self.config_params['irradiance_file'], self.config_params['start'], self.config_params['lapse'])

        # Históricos
        self.history = {'env_param': self.config_params, 'time': [], 'action': [], 'output': [],
                        'irradiance': [], 'inlet_temperature': [], 'ambient_temperature': [],
                        'geometric_efficiency': [], 'outlet_power': [], 'inlet_power': [], 'thermal_power': []}

        # Valores iniciales
        self.state = self.observation_space.sample()
        self.action = np.array([0], dtype=np.float32)
        self.output = self.state[0]

        self.state = np.array([-1] * self.observation_space.shape[0])
        self.action = np.array([0], dtype=np.float32)
        self.output = (self.ACUREXPlant.max_outlet_temp + self.ACUREXPlant.min_outlet_temp) / 2

        # Número de iteraciones en la simulación
        self.step_count = 0

        # "Reward", "done" y "score"
        self.reward = 0
        self.score = 0
        self.done = False

        # Gráficas
        self.fig = None
        self.axs = None
        self.ims = [None for _ in range(7)]

    def reset(self, **kwargs):
        """
        Reset del environment.

        Se devuelven todos los atributos del escenario a su estado inicial.

            · Entradas:
                - kwargs: Diccionario de parámetros: irradiance_file, start, lapse.

            · Salidas:
                - norm_state: Estado inicial del nuevo escenario normalizado.
        """

        # Configuración de la simulación
        self.config_params = {'irradiance_file': 0, 'start': 0, 'lapse': 0}
        if len(kwargs) > 0:
            for param, value in kwargs.items():
                if param in self.config_params.keys():
                    self.config_params[param] = value

        self.start_time, self.stop_time = 9, 18
        self.current_date = {'day': 16, 'month': 11, 'year': 1998}
        self.irradiance_profile = []
        self.Tsim = 1

        self.load_file(self.config_params['irradiance_file'], self.config_params['start'], self.config_params['lapse'])

        # Histórico
        self.history = {'env_param': self.config_params, 'time': [], 'action': [], 'output': [],
                        'irradiance': [], 'inlet_temperature': [], 'ambient_temperature': [],
                        'geometric_efficiency': [], 'outlet_power': [], 'inlet_power': [], 'thermal_power': []}

        # Valores iniciales
        self.state = self.observation_space.sample()
        self.action = np.array([0], dtype=np.float32)
        self.output = self.state[0]

        self.state = np.array([-1] * self.observation_space.shape[0])
        self.action = np.array([0], dtype=np.float32)
        self.output = (self.ACUREXPlant.max_outlet_temp + self.ACUREXPlant.min_outlet_temp) / 2

        # Número de iteraciones en la simulación
        self.step_count = 0

        # "Reward", "done" y "score
        self.reward = 0
        self.score = 0
        self.done = False

        # Gráficas
        plt.close()
        self.fig = None
        self.axs = None
        self.ims = [None for _ in range(len(self.ims))]

        return self.normalize(self.state, self.observation_space.high, self.observation_space.low), {}

    def render(self, mode='human'):
        """ Representación del resultado de la simulación. """

        # Representación inicial
        if self.fig is None:
            plt.ion()
            self.fig, self.axs = plt.subplots(3, 1, figsize=(20, 10), gridspec_kw={'height_ratios': [2, 1, 2]})

            self.axs[0].set_title('Evolución del Sistema')
            self.ims[0], = self.axs[0].plot(self.history['output'])
            self.ims[1], = self.axs[0].plot(self.history['thermal_power'])
            self.axs[0].legend(['$T_{out}$ [ºC]', 'P [kW]'])
            self.axs[0].set_xlim([self.start_time, self.stop_time])
            self.axs[0].set_ylim([0, 500])
            self.axs[0].set_xlabel('Time [h]')

            self.axs[1].set_title('Acción de control')
            self.ims[2], = self.axs[1].plot(self.history['action'])
            self.axs[1].legend(['q [l/s]'])
            self.axs[1].set_xlim([self.start_time, self.stop_time])
            self.axs[1].set_ylim([0, 1.5])
            self.axs[1].set_xlabel('Time [h]')

            self.axs[2].set_title('Perturbaciones')
            self.ims[3], = self.axs[2].plot(self.history['inlet_temperature'])
            self.ims[4], = self.axs[2].plot(self.history['ambient_temperature'])
            self.ims[5], = self.axs[2].plot(self.history['irradiance'])
            self.ims[6], = self.axs[2].plot(self.history['geometric_efficiency'])
            self.axs[2].legend(['Irradiance', '$T_{in}$', '$T_{amb}$', '$\eta_o$ * 500'])
            self.axs[2].set_xlim([self.start_time, self.stop_time])
            self.axs[2].set_ylim([0, 1500])
            self.axs[2].set_xlabel('Time (h)')

            plt.subplots_adjust(hspace=0.5)

        # Actualización de la información
        else:

            x = self.history['time']
            for i, variable in zip(range(len(self.ims)),
                                   [self.history['output'][:-1],
                                    [i/1000 for i in self.history['thermal_power']],
                                    self.history['action'],
                                    self.history['irradiance'],
                                    self.history['inlet_temperature'],
                                    self.history['ambient_temperature'],
                                    [i * 500 for i in self.history['geometric_efficiency']]]):
                self.ims[i].set_xdata(x if len(x) == len(variable) else x[:-1])
                self.ims[i].set_ydata(variable)

        # plt.pause(0.0001)

        return {}

    def step(self, desired_action):
        """
        Paso de simulación.

        A partir del estado actual del escenario y la acción realizada sobre él, se obtiene el siguiente estado del
        mismo. En realidad, dada la discrepancia entre el periodo de control y el periodo de simulación, se realizan
        tantos pasos de simulación como sean necesarios para alcanzar el periodo de control (nsim_per_control).

            · Entradas:
                - desired_action: Acción de control realizada sobre el escenario.
            · Salidas:
                - norm_state: Estado resultante, normalizado, tras la acción realizada.
                - reward: Recompensa obtenida por dicha acción (dada la situación en la que se encontraba el escenario).
                - done: Condición de finalización de la simulación.
        """

        # Saturación de la acción de control
        desired_action = np.array(desired_action).reshape(-1, )
        self.action = np.clip(desired_action, self.low_action, self.high_action)[0]

        logger.debug('[' + str(self.step_count) + '] ' +
                     'Desired Action: ' + str(desired_action) +
                     '; Clipped Action: ' + str(self.action))

        # Simulación hasta siguiente instante de control
        irr, in_temp, amb_temp, no = 0, 0, 0, 0
        for i in range(self.nsim_per_control):

            # Instante t_k = t_{k-1} + dt
            self.step_count += 1

            # Datos de entrada
            irr, in_temp, amb_temp, no = self.read_disturbances(self.step_count)

            outlet_power = self.ACUREXPlant.calculate_thermal_power(self.output, self.action/1000)          # kW
            inlet_power = self.ACUREXPlant.calculate_thermal_power(in_temp, self.action/1000)               # kW
            net_power = outlet_power - inlet_power

            # Históricos de datos
            self.history['time'].append(self.step_to_time(self.step_count))
            self.history['action'].append(self.action)
            self.history['output'].append(self.output)
            self.history['irradiance'].append(irr)
            self.history['inlet_temperature'].append(in_temp)
            self.history['ambient_temperature'].append(amb_temp)
            self.history['geometric_efficiency'].append(no)
            self.history['outlet_power'].append(outlet_power)
            self.history['inlet_power'].append(inlet_power)
            self.history['thermal_power'].append(net_power)

            # Predicción de salida (t_{k+1} = t_k + dt)
            output = self.ACUREXPlant.predict(self.output, self.action, irr, in_temp, amb_temp, no)
            self.output = output

            if self.step_count >= self.Tsim:
                break

        logger.debug('[' + str(self.step_count) + '] ' +
                     'Out. Temp.: ' + str(self.output) +
                     '; Irr.: ' + str(irr) +
                     '; In. Temp.: ' + str(in_temp) +
                     '; Amb. Temp.: ' + str(amb_temp) +
                     '; Geom. Eff.: ' + str(no))

        # Siguiente estado
        self.state = self.build_state()
        norm_state = self.normalize(self.state, self.observation_space.high, self.observation_space.low)

        # Recompensa
        self.reward = self.reward_function()

        # Fin del episodio
        self.done = self.check_end()

        return norm_state, self.reward, self.done, False, {}

    def build_state(self):
        """
        Construcción del estado del escenario a partir de histórico de datos y predicciones de perturbaciones.

        · Salidas:
            - state: Estado del escenario.
        """

        # Parámetros de tiempo
        k = self.step_count
        dt = self.nsim_per_control

        # Construcción del estado
        # -> y(k), ..., y(k-Dy)
        Tout, Wout, Win = [], [], []
        for i in range(self.n_past_outputs + 1):
            Tout += [self.history['output'][k - i * dt - 1] if k - i * dt - 1 >= 0 else -1]
            Wout += [self.history['outlet_power'][k - i * dt - 1] if k - i * dt - 1 >= 0 else -1]
            Win += [self.history['inlet_power'][k - i * dt - 1] if k - i * dt - 1 >= 0 else -1]

        # -> u(k), ..., u(k-Du)
        u = []
        for i in range(self.n_past_actions + 1):
            u += [self.history['output'][k - i * dt - 1] if k - i * dt - 1 >= 0 else -1]

        # -> d(k+Ddf), ..., d(k+1)
        I_f, Tin_f, Tamb_f, no_f = [], [], [], []
        for i in range(self.n_future_disturbances, 0, -1):
            disturbances = self.read_disturbances(k + i * dt)
            I_f += [disturbances[0]]
            Tin_f += [disturbances[1]]
            Tamb_f += [disturbances[2]]
            no_f += [disturbances[3]]

        # -> d(k), ... d(k-Ddp)
        I_p, Tin_p, Tamb_p, no_p = [], [], [], []
        for i in range(self.n_past_disturbances + 1):
            I_p += [self.history['irradiance'][k - i * dt - 1] if k - i * dt - 1 >= 0 else -1]
            Tin_p += [self.history['inlet_temperature'][k - i * dt - 1] if k - i * dt - 1 >= 0 else -1]
            Tamb_p += [self.history['ambient_temperature'][k - i * dt - 1] if k - i * dt - 1 >= 0 else -1]
            no_p += [self.history['geometric_efficiency'][k - i * dt - 1] if k - i * dt - 1 >= 0 else -1]

        state = np.array(Tout + u + I_f + I_p + Tin_f + Tin_p + Tamb_f + Tamb_p + no_f + no_p + Wout + Win)

        return state

    def reward_function(self):
        """
        Función de recompensa.

        Cálculo de la recompensa obtenida por el agente en función del binomio (estado, acción). Esto es, dado un estado
        en el que se encuentra el escenario, la ejecución de una acción de control u otra genera una recompensa acorde
        a la idoneidad de la misma. En este caso, la función de recompensa resulta:

           rew = -J = -( -W + psi * (max(Tout-Tout_max, Tout_min - Tout, 0)/Tout_max)**2 + eps * (q(k) - q(k-1))**2 )

           · Salidas:
                - rew: Valor de la recompensa.
        """

        # Parámetros de ajuste
        psi = 1e7
        eps = 1e5

        # Datos de entrada
        flow = self.history['action'][-1]
        last_flow = self.history['action'][- (self.nsim_per_control + 1)] \
            if (self.nsim_per_control + 1) <= len(self.history['action']) else 0

        outlet_temp = self.history['output'][-1]
        outlet_power = self.ACUREXPlant.calculate_thermal_power(outlet_temp, flow/1000)

        inlet_temp = self.history['inlet_temperature'][-1]
        inlet_power = self.ACUREXPlant.calculate_thermal_power(inlet_temp, flow/1000)

        # Reward
        net_power = outlet_power - inlet_power
        output_penalty = (max(outlet_temp - self.ACUREXPlant.max_outlet_opt_temp,
                              self.ACUREXPlant.min_outlet_opt_temp - outlet_temp,
                              0) / self.ACUREXPlant.max_outlet_opt_temp) ** 2
        action_penalty = (flow - last_flow) ** 2

        rew = (net_power - psi * output_penalty - eps * action_penalty) * 1e-5

        self.score += rew

        logger.debug('[' + str(self.step_count) + '] ' +
                     'Tout: ' + str(outlet_temp) + '; Wout: ' + str(outlet_power) +
                     '; Tin: ' + str(inlet_temp) + '; Win: ' + str(inlet_power))

        logger.debug('[' + str(self.step_count) + '] ' +
                     'Net Power: ' + str(net_power) +
                     '; Output Penalty: ' + str(psi * output_penalty) +
                     '; Action Penalty: ' + str(eps * action_penalty))

        return rew

    def check_end(self):
        """
        Finalización de la simulación.

        Comprobación del estado de la simulación para ver si ha de darse por finalizada o no. Se establecen las
        siguientes condiciones:
            a) Fin de tiempo de simulación.
            b) Sobreenfriamiento del líquido caloportador.
            c) Sobrecalentamiento del líquido caloportador.

            · Salidas:
                - done: Condición de parada de la simulación.
        """

        # Condiciones de finalización
        conds = list()
        conds.append(self.step_count >= self.Tsim)                      # Fin de tiempo de simulación
        conds.append(self.output <= self.ACUREXPlant.min_outlet_temp)   # Sobreenfriamiento del líquido caloportador
        conds.append(self.output >= self.ACUREXPlant.max_outlet_temp)   # Sobrecalentamiento del líquido caloportador

        return any(conds)

    def load_file(self, irradiance_file, start, lapse):
        """
        Lectura de datos de entrada al modelo desde fichero.

        Se cargan los datos de irradiancia sobre la instalación, así como los datos de fecha y hora en las que se sitúa
        la simulación.

            · Entradas:
                - irradiance_file: Numeración del archivo de datos leído: perfilX.mat
                - start: Hora de inicio de la ventana temporal leída.
                - lapse: Duración de la ventana temporal.
        """

        self.irradiance_profile = [0]
        while (np.array(self.irradiance_profile) <= 100).any():     # Se eliminan escenarios donde no hay irradiancia

            # Datos de entrada
            self.config_params['irradiance_file'] = rn.randint(1, 100) if not irradiance_file else irradiance_file
            self.config_params['start'] = rn.uniform(9, 18) if not start else start
            self.config_params['lapse'] = rn.uniform(0.25, 1) if not lapse else lapse

            irradiance_file = None

            # Lectura del fichero
            data_file = scipy.io.loadmat('../data/test' + '/perfil' + str(self.config_params['irradiance_file']) + '.mat')

            # Ventana temporal
            time = data_file['perfil'][:, 0]

            time_start = self.config_params['start']
            idx_start = int(np.where(abs(time - time_start) == min(abs(time - time_start)))[0][0])
            self.start_time = float(time[idx_start])

            time_stop = time_start + self.config_params['lapse']
            future_steps = self.dt_disturbances * self.nsim_per_sample * self.n_future_disturbances
            idx_stop = int(np.where(abs(time - time_stop) == min(abs(time - time_stop)))[0][0]) + future_steps
            self.stop_time = float(time[idx_stop - future_steps])

            # Datos
            self.irradiance_profile = self.interpolate(self.nsim_per_sample, data_file['perfil'][idx_start:idx_stop, 1])

            self.current_date = {'day': int(data_file['perfil'][0, 2]),
                                 'month': int(data_file['perfil'][0, 3]),
                                 'year': int(data_file['perfil'][0, 4])}

            self.Tsim = len(self.irradiance_profile) - future_steps * self.nsim_per_sample

    def read_disturbances(self, step):
        """
        Cálculo de perturbaciones del sistema.

            · Entrada:
                - step: Instante en el que se evalúan las perturbaciones.
            · Salidas:
                - irr: Irradiancia sobre la instalación.
                - in_temp: Temperatura de entrada del fluido al conducto.
                - amb_temp: Temperatura ambiente.
                - no: Eficiencia geométrica dada por el día del año.
        """

        # Irradiancia
        irr = self.irradiance_profile[step]

        # Temperatura de entrada
        last_in_temp = self.history['inlet_temperature'][-1] if self.history['inlet_temperature'] \
            else (self.ACUREXPlant.min_inlet_temp + self.ACUREXPlant.max_inlet_temp) / 2

        in_temp = (600 - self.sim_time) / 600 * last_in_temp + self.sim_time / 600 * (self.output - 90)

        # Temperatura ambiente
        amb_temp = self.history['ambient_temperature'][-1] if self.history['ambient_temperature'] \
            else 25 + 2 * rn.random()

        # Eficiencia geométrica
        no = self.ACUREXPlant.calculate_geometric_efficiency(self.step_to_time(step), self.current_date)

        return irr, in_temp, amb_temp, no

    def step_to_time(self, step):
        """
        Cálculo de la hora del día simulada a partir del instante de simulación.

            · Entradas:
                - step: Instante de simulación.
            · Salidas:
                - hour: Hora del día.
        """

        return self.start_time + (self.sim_time / 3600) * step

    def time_to_step(self, hour):
        """
        Cálculo del instante de simulación a partir de la hora del día simulada.

            · Entradas:
                - hour: Hora del día.
            · Salidas:
                - step: Instante de simulación.
        """

        return int((hour - self.start_time) * 3600 / self.sim_time)

    def save_history(self, path):
        """ Guardado de datos de simulación. """

        with open(path + '/env_history.pickle', 'wb') as f:
            pickle.dump(self.history, f, pickle.HIGHEST_PROTOCOL)

    def load_history(self, path):
        """ Lectura de datos de simulación. """

        with open(path + '/env_history.pickle', 'rb') as f:
            self.history = pickle.load(f)

    @staticmethod
    def interpolate(n_samples, in_vec):
        """
        Interpolación de los elementos de un vector.

        Esta función permite, a partir de un vector de origen, generar un vector de salida con (n_samples-1) muestras
        entre cada pareja de elementos del vector original (utilizando interpolación lineal).

        Por ejemplo:
            >> interpolate(4, [1, 2, 3])
            >> [1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3]

            · Entradas:
                - n_samples: número de puntos intermedios tomados entre dos elementos del vector origen
                - in_vec: vector origen
            · Salidas:
                - out_vec = vector de salida
        """
        out_vec = np.arange(in_vec[0], in_vec[1], (in_vec[1] - in_vec[0]) / n_samples).reshape(-1, ).tolist()

        for i in range(1, len(in_vec) - 1):
            vec_aux = np.linspace(in_vec[i], in_vec[i + 1], n_samples + 1).tolist()
            out_vec.extend(vec_aux[:-1])

        out_vec.append(in_vec[-1])
        out_vec = list(map(float, out_vec))

        return out_vec

    @staticmethod
    def normalize(v, vmax, vmin):
        """
        Normalización de los componentes de un vector al rango (-1, 1).

        · Entradas
            - v: vector a normalizar
            - vmax: vector con valores máximos alcanzables en dicho vector
            - vmin: vector con valores mínimos alcanzables en dicho vector
        · Salidas
            - w: vector normalizado
        """

        return 2 * (v - vmin)/(vmax - vmin) - 1
