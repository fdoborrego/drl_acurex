import logging
import numpy as np
import torch as t
import torch.nn.functional as f
from utils.buffer import ReplayBuffer
from utils.noise import OUActionNoise
from networks.DDPGNetworks import ActorNetwork, CriticNetwork
# from utils.clipper import AutoClipper


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


class Agent:
    """
    Agente Deep Deterministic Policy Gradient:
        · Algoritmo Actor-Critic: Critic estima función acción-estado, y Actor actualiza política en esa dirección.
        · Model-free: no requiere una función de probabilidad asociada a la transición entre estados.
        · Off-policy: se actualiza la política sin usar la política actual para ello.
        · Espacio de acción continuo.
    """

    def __init__(self,
                 env,                                   # Environment
                 lr_actor=1e-4,                         # Learning rate de la Actor nn
                 lr_critic=1e-4,                        # Learning rate de la Critic nn
                 gamma=0.99,                            # Factor de descuento de la recompensa
                 tau=0.001,                             # Factor de filtrado para actualizar la target nn (Polyak)
                 fc1_dims=400,                          # Tamaño de la Hidden Dense Layer 1
                 fc2_dims=300,                          # Tamaño de la Hidden Dense Layer 2
                 mem_size=1000000,                      # Tamaño del buffer de memoria
                 batch_size=64                          # Número de muestras tomadas del Buffer para aprenderla
                 ):

        # Parámetros de entrada
        self.env = env
        input_dims = env.observation_space.shape
        n_actions = env.action_space.shape[0]

        self.lr_critic = lr_critic
        self.lr_actor = lr_actor
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size

        self.action_scale = env.action_space.high

        # Replay Buffer para almacenar transiciones
        self.memory = ReplayBuffer(input_shape=input_dims, n_actions=n_actions, max_size=mem_size)

        # Ruido Ornstein-Uhlenbeck
        self.noise = OUActionNoise(mu=np.zeros(n_actions), sigma=0.1)

        # Redes Neuronales
        self.actor = ActorNetwork(input_dims, n_actions, lr_actor, self.action_scale, fc1_dims, fc2_dims, name='actor')
        self.critic = CriticNetwork(input_dims, n_actions, lr_critic, fc1_dims, fc2_dims, name='critic')
        self.target_actor = ActorNetwork(input_dims, n_actions, lr_actor, self.action_scale, fc1_dims, fc2_dims,
                                         name='target_actor')
        self.target_critic = CriticNetwork(input_dims, n_actions, lr_critic, fc1_dims, fc2_dims, name='target_critic')

        self.update_network_parameters(tau=1)

        # Autoclipper (para evitar exploding gradients)
        # self.actor_clipper = AutoClipper(self.actor, clip_percentile=10, history_size=mem_size)

    def choose_action(self, observation, eval_mode=False):
        """ Política del Agente: elige una acción a partir del estado en que se encuentra. """
        # Actor en modo evaluación: evita que se modifiquen los parámetros del layer normalization (media y desv. típ.)
        self.actor.eval()

        # Conversión a tensor
        state = t.tensor(observation[np.newaxis], dtype=t.float).to(self.actor.device)

        # Cálculo de la acción siguiendo una política determinística
        mu = self.actor.forward(state).to(self.actor.device)

        # Ruido a la acción para la exploración (solo en entrenamiento)
        if not eval_mode:
            mu_prime = mu + t.tensor(self.noise(), dtype=t.float).to(self.actor.device)
        else:
            mu_prime = mu

        # Actor en modo training: sí se modifican los parámetros del layer normalization
        self.actor.train()

        return mu_prime.cpu().detach().numpy()[0]   # Conversión tensor → np array

    def remember(self, state, action, reward, next_state, done):
        """ Almacenamiento de transición en el Replay Buffer. """
        self.memory.add_exp(state, action, reward, next_state, done)

    def save(self, path):
        """ Guarda las redes neuronales del Agente. """
        self.actor.save_checkpoint(path)
        self.target_actor.save_checkpoint(path)
        self.critic.save_checkpoint(path)
        self.target_critic.save_checkpoint(path)

    def load(self, path):
        """ Carga las redes neuronales del Agente. """
        self.actor.load_checkpoint(path)
        self.target_actor.load_checkpoint(path)
        self.critic.load_checkpoint(path)
        self.target_critic.load_checkpoint(path)

    def learn(self):
        """ Función de entrenamiento del agente: actualización de las redes neuronales siguiendo el algoritmo DDPG. """
        # Si no hay suficientes experiencias para aprender de ellas, se espera a que las haya
        if self.memory.mem_ptr < self.batch_size:
            return

        # Se toman "batch_size" muestras aleatorias del Replay Buffer para aprender de ellas
        states, actions, rewards, next_states, done = self.memory.sample_exp(self.batch_size)

        # Conversión a tensor
        states = t.tensor(states, dtype=t.float).to(self.actor.device)
        next_states = t.tensor(next_states, dtype=t.float).to(self.actor.device)
        actions = t.tensor(actions, dtype=t.float).to(self.actor.device)
        rewards = t.tensor(rewards, dtype=t.float).to(self.actor.device)
        done = t.tensor(done).to(self.actor.device)

        # Cálculo del valor estado-acción: yi = ri + gamma * Q′(si + 1, μ′(si+1 | θμ′) | θQ′ )
        target_actions = self.target_actor.forward(next_states)                         # μ′(si+1 | θμ′)

        next_critic_value = self.target_critic.forward(next_states, target_actions)     # Q′(si+1, μ′(si+1|θμ′) | θQ′)
        next_critic_value[done] = 0.0                                                   # Si done = 1 -> Q′ = 0
        next_critic_value = next_critic_value.view(-1)                                  # Se elimina 1 dimensión

        target = rewards + self.gamma * next_critic_value                               # yi
        target = target.view(self.batch_size, 1)                                        # Se añade batch dimension

        # Actualización de la Critic Network mediante la minimización de la función de pérdidas:
        #                           L = 1/N * sum(yi -  Q(si,ai|θQ) )^2
        critic_value = self.critic.forward(states, actions)                             # Q(si,ai|θQ)

        self.critic.optimizer.zero_grad()                                               # Inicialización del gradiente
        critic_loss = f.mse_loss(target, critic_value)                                  # Cálculo de L
        critic_loss.backward()                                                          # Back propagation de L
        self.critic.optimizer.step()                                                    # Actualización de la red

        # Actualización de la Actor Network buscando máxima pendiente del gradiente de J (gradient ascent, de ahí el -):
        #                           ∇J ≈ 1/N · sum(  ∇aQ(s, a|θQ)|s=si,a=μ(si) · ∇θμ μ(s|θμ)|si  )
        #                              ≈ 1/N · sum(  ∇θμQ(s,a|θQ)|s=st,a=μ(st|θμ)  )
        self.actor.optimizer.zero_grad()

        action = self.actor.forward(states)                                             # at
        actor_loss = -self.critic.forward(states, action)                               # ∇θμQ(s,a|θQ)|s=st,a=μ(st|θμ)
        actor_loss = t.mean(actor_loss)                                                 # 1/N · sum(...)
        actor_loss.backward()                                                           # Back propagation
        # self.actor_clipper.clip_gradient()
        self.actor.optimizer.step()                                                     # Actualización de la red

        # Copia de valores de las Critic y Actor Network en las target networks
        self.update_network_parameters()

    def update_network_parameters(self, tau=None):
        """ Actualización de las target networks a partir de las Actor y Critic Networks. """
        # Valor por defecto
        if tau is None:
            tau = self.tau

        # Obtención de parámetros de las redes
        actor_params = self.actor.named_parameters()
        critic_params = self.critic.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_params = self.target_critic.named_parameters()

        # Conversión de estos parámetros a un diccionario
        critic_state_dict = dict(critic_params)
        actor_state_dict = dict(actor_params)
        target_critic_state_dict = dict(target_critic_params)
        target_actor_state_dict = dict(target_actor_params)

        # Actualización de parámetros de las target networks
        for name in critic_state_dict:
            critic_state_dict[name] = tau * critic_state_dict[name].clone() + \
                                      (1 - tau) * target_critic_state_dict[name].clone()
        for name in actor_state_dict:
            actor_state_dict[name] = tau * actor_state_dict[name].clone() + \
                                     (1 - tau) * target_actor_state_dict[name].clone()

        # Se cargan los parámetros actualizados de las target networks
        self.target_critic.load_state_dict(critic_state_dict)
        self.target_actor.load_state_dict(actor_state_dict)
