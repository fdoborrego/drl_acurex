import os
import time
import pickle
import logging
import numpy as np
import matplotlib.pyplot as plt
from agents.DDPGAgent import Agent
from environments.ACUREXEnv import ACUREXEnv
from utils.graphics import plot_learning_curve
from torch.utils.tensorboard import SummaryWriter


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


if __name__ == '__main__':

    # Archivo de guardado
    save_name = '2022.10.17'

    # Configuración de directorios
    save_dir = '../saves'
    os.makedirs(save_dir, exist_ok=True)
    save_dir = save_dir + '/' + save_name
    os.makedirs(save_dir, exist_ok=True)

    tensorboard_dir = save_dir + '/tensorboard'
    os.makedirs(tensorboard_dir, exist_ok=True)

    best_dir = save_dir + '/best'
    ckpt_dir = save_dir + '/ckpt'
    os.makedirs(best_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    best = {'root': best_dir, 'nn': best_dir + '/nn', 'env': best_dir + '/env'}
    for key in best:
        os.makedirs(best[key], exist_ok=True)

    ckpt = {'root': ckpt_dir, 'nn': ckpt_dir + '/nn', 'env': ckpt_dir + '/env'}
    for key in ckpt:
        os.makedirs(ckpt[key], exist_ok=True)

    save_paths = {'root': save_dir, 'tensorboard': tensorboard_dir, 'best': best, 'ckpt': ckpt}

    # Escenario y agente
    env = ACUREXEnv()
    env.reset()

    agent = Agent(env=env,
                  lr_actor=1e-4,
                  lr_critic=1e-4,
                  tau=0.001,
                  gamma=0.95,
                  batch_size=128)

    # Parámetros de la simulación
    N_EPISODES = 5000                                               # Simulaciones a ejecutar
    SAVE_EVERY = 100                                                # Guardado automático cada SAVE_EVERY iteraciones

    filename = 'ACUREXEnv-' + \
               'lractor_' + str(agent.lr_actor) + '_lrcritic_' + str(agent.lr_critic) + \
               'tau' + str(agent.tau) + 'gamma' + str(agent.gamma) + '-' + \
               str(N_EPISODES) + 'games'
    figure_file = save_paths['root'] + '/' + filename + '.png'
    writer = SummaryWriter(save_paths['tensorboard'])

    # Entrenamiento
    best_score = env.reward_range[0]
    score_history = []
    avg_score_history = []

    logger.info(' --- [' + save_name + '] --- ')
    for episode in range(N_EPISODES):

        # Inicialización del entorno
        start = 15      # 5
        lapse = 3       # 23.5

        observation, _ = env.reset(start=start, lapse=lapse)

        done = False
        score = 0
        step_count = 0

        # Inicialización del agente
        agent.noise.reset()

        # Entrenamiento de un episodio
        tic = time.process_time()
        while not done:
            # Control: acción elegida por el agente
            action = agent.choose_action(observation)

            # Se ejecuta dicha acción en el entorno
            next_state, reward, done, _, info = env.step(action)

            # Se acumula la recompensa obtenida
            score += reward

            # Se almacena la experiencia en el buffer replay
            agent.remember(observation, action, reward, next_state, done)

            # Paso de entrenamiento
            agent.learn()

            # Actualización de la observación del estado
            observation = next_state
            step_count += 1

        toc = time.process_time()

        # Análisis de los datos del episodio
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        avg_score_history.append(avg_score)

        logger.info('Episode ' + str(episode) + ' (' + str(step_count) + ' steps, ' + str(round(toc - tic, 5)) +
                    ' seconds)->' + ' Score: ' + str(score) + '. Average Score: ' + str(avg_score))

        # Publicación en tensorboard
        writer.add_scalar('Reward/Episodic Reward', score, global_step=episode)

        if episode >= 100:

            # Guardado
            if episode % SAVE_EVERY == 0:
                logger.info('*** Saving checkpoint... Episode ' + str(episode) + ' reached. ***')

                agent.save(save_paths['ckpt']['nn'])
                env.save_history(save_paths['ckpt']['env'])
                with open(save_paths['ckpt']['env'] + '/score_history.pickle', 'wb') as f:
                    pickle.dump(score_history, f, pickle.HIGHEST_PROTOCOL)
                with open(save_paths['ckpt']['env'] + '/avg_score_history.pickle', 'wb') as f:
                    pickle.dump(avg_score_history, f, pickle.HIGHEST_PROTOCOL)

            # Actualización de mejor puntuación
            if avg_score > best_score:
                best_score = avg_score
                logger.info('*** Saving best score... New highest score reached: ' + str(best_score) + '. ***')

                agent.save(save_paths['best']['nn'])
                env.save_history(save_paths['best']['env'])
                with open(save_paths['best']['env'] + '/score_history.pickle', 'wb') as f:
                    pickle.dump(score_history, f, pickle.HIGHEST_PROTOCOL)
                with open(save_paths['best']['env'] + '/avg_score_history.pickle', 'wb') as f:
                    pickle.dump(avg_score_history, f, pickle.HIGHEST_PROTOCOL)

    # Resultado del entrenamiento
    plot_learning_curve(score_history, figure_file)

    # Cierre de escritor de tensorboard
    writer.flush()
    writer.close()
