import numpy as np


class OUActionNoise:
    """
    Ornstein-Uhlenbeck process:
    El proceso de Ornstein-Uhlenbeck es un proceso estacionario de Gauss-Markov, lo que significa que es: un proceso
    gaussiano, un proceso de Markov y temporalmente homogéneo.

    Este proceso permite generar una exploración temporalmente correlada idónea para sistemas de control físicos con
    inercia. En concreto, permitirá obtener una política de exploración (mu') a partir de la política del "actor" (mu)
    tal que:

                                            mu'(state) = mu(state) + N

    , siendo N el ruido del proceso.
    """

    def __init__(self,
                 mu,                                # Media de la distribución
                 sigma=0.30,                        # Desviación típica
                 theta=0.15,                        # Parámetro: f(x_t,t) = x_t * e^(theta * t). Tamaño de paso.
                 dt=1e-2,                           # Variación temporal (para definir la correlación temporal)
                 x0=None):                          # Punto de partida

        self.mu = mu
        self.sigma = sigma
        self.theta = theta
        self.dt = dt
        self.x0 = x0
        self.x_prev = 0
        self.reset()                                # Reiniciamos el ruido

    def __call__(self):
        """ Función __call__ (método que se ejecuta al utilizar la propia clase como función): correlación temporal. """

        # Correlación temporal del ruido
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x

        return x

    def reset(self):
        """ Reinicia el ruido: devuelve el valor de x_prev a su valor inicial: x0. """

        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)
