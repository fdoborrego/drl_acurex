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
    load_name = 'DDPG'
    evaluate = True

    # Creación del escenario y agente
    env = ACUREXEnv()
    eval_env = ACUREXEnv()
    agent = Agent(env=env,
                  lr_actor=1e-4,
                  lr_critic=1e-4,
                  tau=0.001,
                  gamma=0.95,
                  fc1_dims=400,
                  fc2_dims=300,
                  batch_size=128)
    agent.load('../saves/' + load_name + '/best/nn')

    metrics = {'RMSE': [], 'Score': {'DDPG': [], 'MPC': []}, 'Time': {'DDPG': [], 'MPC': []}}

    # Simulación
    N_EPISODES = 3
    for e in range(N_EPISODES):

        # Parámetros de simulación
        irradiance_file = rn.randint(1, 100)
        start = 13
        lapse = 5

        # Inicialización del escenario
        d, R = False, 0
        s, _ = env.reset(irradiance_file=irradiance_file, start=start, lapse=lapse)
        env.render()

        # Simulación
        tic = time.process_time()
        while not d:
            a = agent.choose_action(s, eval_mode=True)
            s, r, d, _, _ = env.step(a)
            R += r
        dt1 = time.process_time() - tic

        # Representación
        env.render()
        plt.savefig("../figures/Sim" + str(e+1) + "_DDPG.png")
        plt.close()

        # Evaluación
        if evaluate:

            eval_env.reset(irradiance_file=irradiance_file, start=start, lapse=lapse)
            eval_agent = MPC(eval_env)

            tic = time.process_time()
            while not eval_env.done:
                action = eval_agent.choose_action(eval_env)
                eval_env.step(action)
            dt2 = time.process_time() - tic

            plot_evaluation({'name': 'DDPG', 'env': env}, {'name': 'MPC', 'env': eval_env})
            plt.savefig("../figures/Sim" + str(e+1) + "_DDPGvsMPC.png")
            plt.close()

            L = min(len(env.history['output']), len(eval_env.history['output']))
            metrics['RMSE'].append(mse(eval_env.history['output'][:L], env.history['output'][:L], squared=False))
            metrics['Score']['DDPG'].append(eval_env.score)
            metrics['Score']['MPC'].append(env.score)
            metrics['Time']['DDPG'].append(dt1)
            metrics['Time']['MPC'].append(dt2)

        env.reset()

    # Métricas
    logger.info('Resultados de la simulación (MPC vs. DDPG): ' +
                '\n · RMSE: ' + str(np.mean(metrics['RMSE'])) +
                '\n · Scores: ' + str(np.mean(metrics['Score']['MPC'])) + ' (MPC) vs '
                                + str(np.mean(metrics['Score']['DDPG'])) + ' (DRLC)' +
                '\n · Times: ' + str(np.mean(metrics['Time']['MPC'])) + ' (MPC) vs. '
                               + str(np.mean(metrics['Time']['DDPG'])) + ' (DRLC)')
