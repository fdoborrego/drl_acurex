import os
import logging
import numpy as np
import torch as t                   # Paquete base
import torch.nn as nn               # Capas de las redes neuronales
import torch.nn.functional as f     # Funciones de activación
import torch.optim as optim         # Funciones de optimización


logger = logging.getLogger(__name__)
logger.disabled = False

logger.setLevel(logging.INFO)  # DEBUG, INFO, WARNING, ERROR, EXCEPTION, CRITICAL
log_format = '[%(levelname)s - %(name)s] (%(asctime)s) %(message)s'
formatter = logging.Formatter(log_format)

consoleh = logging.StreamHandler()
consoleh.setFormatter(formatter)
logger.addHandler(consoleh)

fileh = logging.FileHandler('../log/logs.log')
fileh.setFormatter(formatter)
logger.addHandler(fileh)


class CriticNetwork(nn.Module):
    """
    Critic Network: red que implementa la parte "Critic" del agente.

    El agente "Critic" es el que permite obtener una estimación del valor estado-acción asociado a un par (s, a). Esta
    estimación se utiliza de tal forma que el agente "Actor" elabora una política de acción de que persiga aquellos
    estados de mayor valor Q(s, a).
    """
    def __init__(self, input_dims, n_actions, lr=1e-4, fc1_dims=400, fc2_dims=300,
                 name='critic', chkpt_dir='../src/networks/tmp/ddpg'):
        super(CriticNetwork, self).__init__()

        # Parámetros de entrada
        self.input_dims = input_dims                                            # Dimensión de datos de entrada a nn
        self.n_actions = n_actions                                              # Dimensión del espacio de acción
        self.fc1_dims = fc1_dims                                                # Dimensión de capa 1 de la nn
        self.fc2_dims = fc2_dims                                                # Dimensión de capa 2 de la nn

        self.name = name                                                        # Nombre de la nn
        self.checkpoint_dir = chkpt_dir                                         # Directorio de guardado
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Arquitectura de la red neuronal
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)                   # Input Layer
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)                      # Dense Layer

        self.bn1 = nn.LayerNorm(self.fc1_dims)                                  # Batch normalization for Input Layer
        self.bn2 = nn.LayerNorm(self.fc2_dims)                                  # Batch normalization for Dense Layer

        self.action_value = nn.Linear(self.n_actions, self.fc2_dims)

        self.q = nn.Linear(self.fc2_dims, 1)                                    # Output Layer (Critic Value, Q)

        # Inicialización de la red
        f1 = 1. / np.sqrt(self.fc1.weight.data.size()[0])
        self.fc1.weight.data.uniform_(-f1, f1)
        self.fc1.bias.data.uniform_(-f1, f1)

        f2 = 1. / np.sqrt(self.fc2.weight.data.size()[0])
        self.fc2.weight.data.uniform_(-f2, f2)
        self.fc2.bias.data.uniform_(-f2, f2)

        f3 = 0.003
        self.q.weight.data.uniform_(-f3, f3)
        self.q.bias.data.uniform_(-f3, f3)

        f4 = 1. / np.sqrt(self.q.weight.data.size()[0])
        self.action_value.weight.data.uniform_(-f4, f4)
        self.action_value.bias.data.uniform_(-f4, f4)

        # Optimizador
        self.optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=0.01)

        # (Ejecución mediante GPU)
        self.device = t.device('cuda:0' if t.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, action):
        """ Feed-forward: pasa un par estado-acción a través de la red. """
        state_value = self.fc1(state)
        state_value = self.bn1(state_value)
        state_value = f.relu(state_value)
        state_value = self.fc2(state_value)
        state_value = self.bn2(state_value)

        action_value = self.action_value(action)

        state_action_value = f.relu(t.add(state_value, action_value))
        state_action_value = self.q(state_action_value)

        return state_action_value

    def save_checkpoint(self, path=None):
        """ Guardado de la red """
        # Dirección de guardado
        if path is None:
            path = self.checkpoint_dir
        checkpoint_file = os.path.join(path, self.name)

        # Guardado
        logger.debug('Saving \')' + self.name + '\' network. ')
        t.save(self.state_dict(), checkpoint_file)

    def load_checkpoint(self, path):
        """ Cargado de la red """
        # Dirección de cargado
        if path is None:
            path = self.checkpoint_dir
        checkpoint_file = os.path.join(path, self.name)

        # Lectura
        logger.debug('Loading \')' + self.name + '\' network. ')
        self.load_state_dict(t.load(checkpoint_file))


class ActorNetwork(nn.Module):
    """
    Actor Network: red que implementa la parte "Actor" del agente.

    El agente "Actor" es el encargado de determinar aquella acción a que permita al agente alcanzar el estado s con
    mayor valor Q(s, a). Es decir, este componente es el encargado de desarrollar la política de acción del agente.
    """
    def __init__(self, input_dims, n_actions, lr=1e-4, action_scale=1, fc1_dims=400, fc2_dims=300,
                 name='actor', chkpt_dir='../src/networks/tmp/ddpg'):
        super(ActorNetwork, self).__init__()

        # Parámetros de entrada
        self.input_dims = input_dims                                            # Dimensión de datos de entrada a nn
        self.fc1_dims = fc1_dims                                                # Dimensión de capa 1 de la nn
        self.fc2_dims = fc2_dims                                                # Dimensión de capa 2 de la nn
        self.n_actions = n_actions                                              # Dimensión del espacio de acción

        self.name = name                                                        # Nombre de la nn
        self.checkpoint_dir = chkpt_dir                                         # Directorio de guardado
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Arquitectura de la red neuronal
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)                   # Input Layer
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)                      # Dense Layer

        self.bn1 = nn.LayerNorm(self.fc1_dims)                                  # Batch normalization for Input Layer
        self.bn2 = nn.LayerNorm(self.fc2_dims)                                  # Batch normalization for Dense Layer

        self.mu = nn.Linear(self.fc2_dims, self.n_actions)              # Output Layer (Actions)

        # Inicialización de la red
        f1 = 1. / np.sqrt(self.fc1.weight.data.size()[0])
        self.fc1.weight.data.uniform_(-f1, f1)
        self.fc1.bias.data.uniform_(-f1, f1)

        f2 = 1. / np.sqrt(self.fc2.weight.data.size()[0])
        self.fc2.weight.data.uniform_(-f2, f2)
        self.fc2.bias.data.uniform_(-f2, f2)

        f3 = 0.003
        self.mu.weight.data.uniform_(-f3, f3)
        self.mu.bias.data.uniform_(-f3, f3)

        # Optimizador
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        # (Ejecución mediante GPU)
        self.device = t.device('cuda:0' if t.cuda.is_available() else 'cpu')
        self.to(self.device)

        self.action_scale = t.FloatTensor(action_scale).to(self.device)  # Valor máx. del espacio de acción

    def forward(self, state):
        """ Feed-forward: pasa un estado a través de la red. """
        x = self.fc1(state)
        x = self.bn1(x)
        x = f.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = f.relu(x)
        x = self.mu(x)
        x = t.sigmoid(x)
        x = x * self.action_scale

        return x

    def save_checkpoint(self, path=None):
        """ Guardado de la red """
        # Dirección de guardado
        if path is None:
            path = self.checkpoint_dir
        checkpoint_file = os.path.join(path, self.name)

        # Guardado
        logger.debug('Saving \')' + self.name + '\' network. ')
        t.save(self.state_dict(), checkpoint_file)

    def load_checkpoint(self, path):
        """ Cargado de la red """
        # Dirección de lectura
        if path is None:
            path = self.checkpoint_dir
        checkpoint_file = os.path.join(path, self.name)

        # Lectura
        logger.debug('Loading \')' + self.name + '\' network. ')
        self.load_state_dict(t.load(checkpoint_file))
