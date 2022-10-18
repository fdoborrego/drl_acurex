import time
import logging
import numpy as np
import random as rn
import matplotlib.pyplot as plt
from agents.DDPGAgent import Agent
from agents.MPCController import MPC
from environments.ACUREXEnv import ACUREXEnv
from sklearn.metrics import mean_squared_error as mse
from utils.graphics import plot_evaluation


# Logger
logger = logging.getLogger(__name__)
logger.disabled = False

logger.setLevel(logging.DEBUG)  # DEBUG, INFO, WARNING, ERROR, EXCEPTION, CRITICAL
log_format = '[%(levelname)s - %(name)s] (%(asctime)s) %(message)s'
formatter = logging.Formatter(log_format)

consoleh = logging.StreamHandler()
consoleh.setFormatter(formatter)
logger.addHandler(consoleh)

fileh = logging.FileHandler('../log/logs.log')
fileh.setFormatter(formatter)
logger.addHandler(fileh)


if __name__ == "__main__":

    # Parámetros
    load_name = 'DDPGv2'
    evaluate = True

    # Creación del escenario y agente
    env = ACUREXEnv()
    eval_env = ACUREXEnv()
    agent = Agent(env=env,
                  lr_actor=1e-4,
                  lr_critic=1e-4,
                  tau=0.001,
                  gamma=0.95,
                  batch_size=128)
    agent.load('../saves/' + load_name + '/best/nn')

    metrics = {'RMSE': [], 'Score': {'DDPG': [], 'MPC': []}, 'Time': {'DDPG': [], 'MPC': []},
               'Power': {'DDPG': [], 'MPC': []}, 'AACI': {'DDPG': [], 'MPC': []}, 'MSCV': {'DDPG': [], 'MPC': []}}

    # Simulación
    N_EPISODES = 3
    for e in range(N_EPISODES):

        # Parámetros de simulación
        irradiance_file = rn.randint(1, 100)
        start = 13
        lapse = 5

        # Inicialización del escenario
        d = False
        s, _ = env.reset(irradiance_file=irradiance_file, start=start, lapse=lapse)
        env.render()

        # Simulación
        rewards1 = []

        tic = time.process_time()
        while not d:
            a = agent.choose_action(s, eval_mode=True)
            s, r, d, _, _ = env.step(a)
            rewards1.append(r)
        dt1 = time.process_time() - tic

        # Representación
        env.render()
        plt.savefig("../figures/Sim" + str(env.config_params['irradiance_file']) + "_DDPG.png")
        plt.close()

        # Evaluación
        if evaluate:

            eval_env.reset(irradiance_file=env.config_params['irradiance_file'],
                           start=env.config_params['start'],
                           lapse=env.config_params['lapse'])
            eval_env.render()
            eval_agent = MPC(eval_env)

            rewards2 = []

            tic = time.process_time()
            while not eval_env.done:
                action = eval_agent.choose_action(eval_env)
                eval_env.step(action)
                rewards2.append(eval_env.reward)
            dt2 = time.process_time() - tic

            eval_env.render()
            plt.savefig("../figures/Sim" + str(env.config_params['irradiance_file']) + "_MPC.png")
            plt.close()

            plot_evaluation({'name': 'DDPG', 'env': env}, {'name': 'MPC', 'env': eval_env})
            plt.savefig("../figures/Sim" + str(env.config_params['irradiance_file']) + "_DDPGvsMPC.png")
            plt.close()

            m = int(30 * 60 / env.sim_time)   # 30 min para estabilizarse
            n = min(len(env.history['output']), len(eval_env.history['output']))
            metrics['RMSE'].append(mse(eval_env.history['output'][m:n], env.history['output'][m:n], squared=False))
            metrics['Score']['MPC'].append(
                np.sum(rewards2[int(m/eval_env.nsim_per_control):int(n/eval_env.nsim_per_control)]))
            metrics['Score']['DDPG'].append(
                np.sum(rewards1[int(m/env.nsim_per_control):int(n/env.nsim_per_control)]))
            metrics['Time']['MPC'].append(
                dt2)
            metrics['Time']['DDPG'].append(
                dt1)
            metrics['Power']['MPC'].append(
                np.mean(eval_env.history['thermal_power'][m:n]))
            metrics['Power']['DDPG'].append(
                np.mean(env.history['thermal_power'][m:n]))
            metrics['AACI']['MPC'].append(
                np.sum(np.array(eval_env.history['action'][m:n])-np.array([0] + eval_env.history['action'][m:n-1])))
            metrics['AACI']['DDPG'].append(
                np.sum(np.array(env.history['action'][m:n])-np.array([0] + env.history['action'][m:n-1])))
            metrics['MSCV']['MPC'].append(
                np.mean(np.maximum.reduce(
                    [np.array(eval_env.history['output'][m:n]) - eval_env.ACUREXPlant.max_outlet_opt_temp,
                     eval_env.ACUREXPlant.min_outlet_opt_temp - np.array(eval_env.history['output'][m:n]),
                     np.zeros_like(np.array(eval_env.history['output'][m:n]))])**2))
            metrics['MSCV']['DDPG'].append(
                np.mean(np.maximum.reduce(
                    [np.array(env.history['output'][m:n]) - env.ACUREXPlant.max_outlet_opt_temp,
                     env.ACUREXPlant.min_outlet_opt_temp - np.array(env.history['output'][m:n]),
                     np.zeros_like(np.array(env.history['output'][m:n]))])**2))

        env.reset()

    # Métricas
    logger.info('Resultados de la simulación (MPC vs. DDPG): ' +
                '\n · RMSE: ' + str(np.mean(metrics['RMSE'])) +
                '\n · Score: ' + str(np.mean(metrics['Score']['MPC'])) + ' (MPC) vs '
                               + str(np.mean(metrics['Score']['DDPG'])) + ' (DRLC)' +
                '\n · Time: ' + str(np.mean(metrics['Time']['MPC'])) + ' (MPC) vs. '
                              + str(np.mean(metrics['Time']['DDPG'])) + ' (DRLC)' +
                '\n · Power: ' + str(np.mean(metrics['Power']['MPC'])) + ' (MPC) vs. '
                               + str(np.mean(metrics['Power']['DDPG'])) + ' (DRLC)' +
                '\n · AACI: ' + str(np.mean(metrics['AACI']['MPC'])) + ' (MPC) vs. '
                              + str(np.mean(metrics['AACI']['DDPG'])) + ' (DRLC)' +
                '\n · MSCV: ' + str(np.mean(metrics['MSCV']['MPC'])) + ' (MPC) vs. '
                              + str(np.mean(metrics['MSCV']['DDPG'])) + ' (DRLC)')
