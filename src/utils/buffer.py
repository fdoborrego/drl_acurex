import numpy as np


class ReplayBuffer:
    """
    Buffer de almacenamiento de transiciones.

    Este buffer permite almacenar las transiciones ya efectuadas por el agente, permitiendo recordar:
        · state: Estado de partida.
        · action: Acción tomada en la transición.
        · next_state: Siguiente estado alcanzado.
        · reward: Recompensa recibida en la transición.
        · done: Flag para indicar si el estado alcanzado es terminal.
    """

    def __init__(self, input_shape, n_actions, max_size=1000000):
        self.mem_size = max_size                                            # Tamaño del buffer
        self.mem_ptr = 0                                                    # Puntero (primera posición vacía)
        self.state_memory = np.zeros((self.mem_size, *input_shape))         # Memoria de: estado
        self.next_state_memory = np.zeros((self.mem_size, *input_shape))    # Memoria de: siguiente estado
        self.action_memory = np.zeros((self.mem_size, n_actions))           # Memoria de: acciones
        self.reward_memory = np.zeros(self.mem_size)                        # Memoria de: rewards
        self.done_memory = np.zeros(self.mem_size, dtype=np.bool)           # Memoria de: flag de finalización (done)

    def add_exp(self, state, action, reward, next_state, done):
        """ Añade una transición al buffer. """

        # Posición de almacenamiento
        index = self.mem_ptr % self.mem_size            # Implementa una cola circular
        self.mem_ptr += 1                               # Actualización del puntero (para la siguiente transición)

        # Almacenamiento de nuevo episodio
        self.state_memory[index] = state
        self.next_state_memory[index] = next_state
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.done_memory[index] = done

    def sample_exp(self, batch_size):
        """ Extrae un número de transiciones del buffer. """

        # Muestras a tomar
        max_mem = min(self.mem_ptr, self.mem_size)      # Máx. mem. disponible -> no tomar más muestras de las que hay
        batch = np.random.choice(max_mem, batch_size)   # Muestreo aleatorio

        # Extracción de experiencias
        states = self.state_memory[batch]
        next_states = self.next_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.done_memory[batch]

        return states, actions, rewards, next_states, dones
