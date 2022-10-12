import numpy as np
import matplotlib.pyplot as plt


def plot_learning_curve(scores):

    scores = np.array(scores[1:])[np.array(scores) > -100]              # Removing outliers
    x = [i + 1 for i in range(len(scores))]

    train_mean = np.zeros(len(scores))
    train_std = np.zeros(len(scores))
    for i in range(len(train_mean)):
        train_mean[i] = np.mean(scores[max(0, i-100):(i + 1)])
        train_std[i] = np.std(scores[max(0, i-100):(i + 1)])

    plt.plot(x, train_mean, color='blue', marker='o', markersize=5, label='Training Accuracy')
    plt.fill_between(x, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')

    plt.xlim([x[0], x[-1]])
    plt.title('Curva de aprendizaje')
    plt.xlabel('Nº de Episodios')
    plt.ylabel('Score')
    plt.grid()
    plt.show()


def plot_evaluation(env1, env2):

    plt.ion()
    fig, axs = plt.subplots(2, 1, figsize=(20, 10), gridspec_kw={'height_ratios': [2, 1]})

    axs[0].set_title('Evolución del Sistema')
    axs[0].plot(env1['env'].history['time'], env1['env'].history['output'])
    axs[0].plot(env1['env'].history['time'], [i / 1000 for i in env1['env'].history['thermal_power']])
    axs[0].plot(env2['env'].history['time'], env2['env'].history['output'])
    axs[0].plot(env2['env'].history['time'], [i / 1000 for i in env2['env'].history['thermal_power']])
    axs[0].legend(['$T_{out}^{' + env1['name'] + '}$', '$P^{' + env1['name'] + '}$',
                   '$T_{out}^{' + env2['name'] + '}$', '$P^{' + env2['name'] + '}$'])
    axs[0].set_xlim([env1['env'].start_time, env1['env'].stop_time])
    axs[0].set_ylim([0, 500])
    axs[0].set_xlabel('Time [h]')

    axs[1].set_title('Acción de Control')
    axs[1].plot(env1['env'].history['time'], env1['env'].history['action'])
    axs[1].plot(env2['env'].history['time'], env2['env'].history['action'])
    axs[1].legend(['$q^{' + env1['name'] + '}$', '$q^{' + env2['name'] + '}$'])
    axs[1].set_xlim([env1['env'].start_time, env1['env'].stop_time])
    axs[1].set_ylim([0, 1.5])
    axs[1].set_xlabel('Time [h]')