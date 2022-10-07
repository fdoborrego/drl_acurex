import logging
import matplotlib.pyplot as plt
from environments.ACUREXEnv import ACUREXEnv
from agents.MPCController import MPC


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

    # Escenario
    env = ACUREXEnv()
    env.reset(irradiance_file=87, start=15, lapse=0.5)
    env.render()

    control = MPC(env)

    # Simulación
    score = 0
    R_vec = [score]
    step = 0

    while not env.done:

        a = control.choose_action(env)
        s, r, d, _, _ = env.step(a)

        score += r
        R_vec.append(score)

        step += 1
        logger.info('Time-step: ' + str(step) + '. Score: ' + str(score) + '.')

    # Representación de resultados
    env.render()
    plt.savefig("../figures/mpc.png")
    plt.waitforbuttonpress(0)

    env.reset()

    logger.info('Recompensa: ' + str(score) + '. Time-steps: ' + str(step + 1))
