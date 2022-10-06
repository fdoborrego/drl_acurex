from math import sin, cos, tan, acos, sqrt, pi


class ACUREXModel:
    """
    Modelo de parámetros concentrados de la planta termosolar ACUREX.
    """

    environment_name = "ACUREX solar field - Concentrated parameters model"

    def __init__(self,
                 integration_time                                      # Tiempo de integración (\Delta T_sim) [s]
                 ):

        # Parámetros de entrada
        self.integration_time = integration_time

        """ Descripción del sistema """
        # Parámetros temporales
        self.sample_time = 1                                        # Periodo de muestreo de sensores [s]
        self.control_time = 30                                      # Periodo del controlador [s]

        # Dimensiones del sistema
        inner_diameter = 0.0254                                     # Diámetro interior del tubo [m]
        external_diameter = 0.031                                   # Diámetro exterior del tubo [m]
        self.fluid_area = pi * inner_diameter**2 / 4                # Área de paso del fluido por el tubo [m2]
        self.pipe_area = pi * (external_diameter**2 -
                               inner_diameter**2) / 4               # Área de corona metálica del tubo [m2]

        self.collector_opening = 1.82                               # Apertura del colector, G [m]
        self.focal_distance = 1.1                                   # Distancia focal [m]

        self.loop_length = 172                                      # Longitud de un lazo de colectores [m]
        self.loop_area = 148 * self.collector_opening               # Área de intercambio del lazo de colectores [m]

        # Posición geográfica de la planta
        self.longitude = -6.265939                                  # Longitud [ª]
        self.latitude = 0.675 * 180 / pi                            # Latitud [º]

        # Eficiencia óptica de un colector
        reflectivity = 0.74
        form_factor = 0.9
        pipe_efficiency = 0.85
        self.optical_efficiency = reflectivity * form_factor * pipe_efficiency

        # Pérdidas de superficie
        self.non_effective_area = 3                                 # Superficie no efectiva
        self.geometric_loss_area = 2 * self.focal_distance \
            * self.collector_opening                                # Superficie perdida por el ángulo de incidencia
        self.collector_area = 37 * self.collector_opening           # Superficie total del colector [m2]

        """ Restricciones """
        self.min_outlet_temp, self.max_outlet_temp = -25, 315                   # [ºC]
        self.min_irradiance, self.max_irradiance = 0, 1.5e3                     # [W/m2]
        self.min_inlet_temp, self.max_inlet_temp = -25, 250                     # [ºC]
        self.min_ambient_temp, self.max_ambient_temp = 0, 40                    # [ºC]
        self.min_geometric_efficiency, self.max_geometric_efficiency = 0, 1     # [p.u.]
        self.min_flow, self.max_flow = 0.2, 1.2                                 # [l/s]

        self.min_outlet_opt_temp, self.max_outlet_opt_temp = 200, 300           # [ºC]

    def predict(self, last_outlet_temp, flow, irradiance, inlet_temp, ambient_temp, geometric_efficiency):
        """
        Modelo de parámetros concentrados: cálculo de la temperatura de salida de un lazo de colectores.

            · Entradas:
                - last_outlet_temp: Temperatura de salida en el último instante de simulación [°C].
                - flow: Caudal de aceite [l/s].
                - irradiance: Irradiancia [W/m2].
                - inlet_temp: Temperatura de entrada al lazo [°C].
                - ambient_temp: Temperatura ambiente [°C].
                - geometric_efficiency: Eficiencia geométrica [p.u.].
            · Salidas:
                - outlet_temp: Temperatura de salida [°C].
        """

        # Parámetros del sistema
        # - Fluido
        mean_temp = (last_outlet_temp + inlet_temp) / 2
        rho, Cf = self.calculate_fluid_properties(mean_temp)

        # - Transmisión de calor
        C = self.loop_length * self.fluid_area * rho * Cf
        Hl = 0.00249 * (mean_temp - ambient_temp) - 0.06133

        # Ecuación diferencial
        # - Temperatura de salida
        derT = (1/C) * (self.optical_efficiency * geometric_efficiency * self.loop_area * irradiance
                        - flow * 1e-3 * rho * Cf * (last_outlet_temp - inlet_temp)
                        - Hl * self.loop_area * (mean_temp - ambient_temp))

        # Integración
        outlet_temp = last_outlet_temp + derT * self.integration_time

        return outlet_temp

    def calculate_thermal_power(self, temp, flow):
        """
        Cálculo de la potencia térmica portada por el fluido.

            · Entrada:
                - temp: Temperatura a la que se calcula la potencia térmica.
                - flow: Caudal de fluido.
            · Salidas:
                - power: Potencia térmica.
        """

        # Propiedades del fluido
        rho, Cf = self.calculate_fluid_properties(temp)

        # Potencia térmica
        power = rho * Cf * flow * temp

        return power

    def calculate_geometric_efficiency(self, hour, date):
        """
        Cálculo de la eficiencia geométrica.

            · Entradas:
                - hour: Hora actual oficial (= la que marca el reloj) (0 - 23) (+ min).
                - date: Fecha ({day: 'XX', month: 'XX', year: 'XXXX'}).
            · Salidas:
                - no: Eficiencia geométrica.
        """

        # Cálculos previos
        # - Día juliano
        julian_day = self.calculate_julian_day(date['day'], date['month'], date['year'])

        # - Ángulo diario
        daily_angle = self.calculate_daily_angle(julian_day)

        # - Hora solar
        solar_time = self.calculate_solar_time(self.longitude, hour, date['month'], daily_angle)

        # Coordenadas solares temporales
        # - Declinación calculada por la fórmula de Spencer
        delta1 = 0.006918 - 0.399912 * cos(daily_angle) + 0.070257 * sin(daily_angle) - \
            0.006758 * cos(2 * daily_angle) + 0.000907 * sin(2 * daily_angle) - \
            0.002697 * cos(3 * daily_angle) + 0.00148 * sin(3 * daily_angle)

        # - Ángulo horario
        delta2 = ((solar_time - 12) * 15) * (pi / 180)

        # Ángulo de incidencia (= ángulo entre el vector solar y la normal a la superficie del colector)
        fi1 = acos(sqrt(1 - cos(delta1)**2 * sin(delta2)**2))

        # Eficiencia geométrica
        no = 1 - (self.non_effective_area + self.geometric_loss_area * abs(tan(fi1))) / self.collector_area
        no = no * (sqrt(1 - cos(delta1)**2 * sin(delta2)**2))

        return no

    @staticmethod
    def calculate_solar_time(long, current_local_hour, month, daily_angle):
        """
        Cálculo de la hora solar (= hora que marca un reloj de sol). Esta hora refleja el movimiento real del Sol.

            · Entradas:
                - long: Longitud geográfica de la posición de la planta.
                - current_local_hour: "Hora actual (= marcada por reloj local)" (+ min).
                - month: Mes del año (1 - 12).
                - daily_angle: Ángulo diario (0 - 2pi).
            · Salidas:
                - solar_time: Hora solar.

        """
        # Hora oficial en GMT (en decimal)
        month_diff = -120 if 4 <= month <= 10 else -60                      # Diferencia horaria [min] con GMT+1/GMT+2
        current_official_time = current_local_hour * 60 + month_diff

        # Diferencia de posición en minutos
        long_central_meridian = 0
        pos_diff = abs(long - long_central_meridian) * 4

        # Ecuación del tiempo (EdT): diferencia entre hora solar y hora oficial corregida en longitud
        EdT = (0.000075 + 0.001868 * cos(daily_angle) - 0.032077 * sin(daily_angle)
               - 0.014615 * cos(2 * daily_angle) - 0.04089 * sin(2 * daily_angle)) * 229.18       # En minutos

        # Hora solar
        solar_time = (current_official_time - pos_diff + EdT)/60

        return solar_time

    @staticmethod
    def calculate_daily_angle(julian_day):
        """
        Cálculo del ángulo barrido por la Tierra en su movimiento de traslación alrededor del Sol desde el día 1 de
        enero.

            · Entradas:
                - julian_day: Día juliano (1-365).
            · Salidas:
                - daily_angle: Ángulo [0 - 2 pi].
        """
        return 2 * pi / 365. * (julian_day - 1)

    @staticmethod
    def calculate_julian_day(day, month, year):
        """
        Cálculo del día juliano (= número de días y fracción transcurridos desde el mediodía del 1 de enero del año
        4713 a.C.). En este caso, solo se calcula el día juliano con respecto al 1 de enero del mismo año considerado.

            · Entradas:
                - day: Día del mes (1 - 31).
                - month: Mes del año (1 - 12).
                - year: Año.
            · Salidas:
                - Día juliano (0-365).
        """

        # Días por cada mes en un año
        days_per_month_noleap = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]        # Año normal (no bisiesto)
        days_per_month_leap = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]          # Año bisiesto

        # Comprobación de año bisiesto
        if (year % 4) == 0 and ((year % 100) != 0 or (year % 400) == 0):
            days_per_month = days_per_month_leap
        else:
            days_per_month = days_per_month_noleap

        # Cálculo del día juliano
        julian_day = day + sum(days_per_month[:month - 1])

        return julian_day

    @staticmethod
    def calculate_fluid_properties(temp):
        """
        Cálculo de las propiedades del fluido.

            · Entradas:
                - temp: Temperatura del fluido [°C].
            · Salidas:
                - rho: Densidad [kg/m3].
                - Cf: Capacidad calorífica específica del fluido [J/kg°C].
        """
        # Datos de entrada
        Tf = temp

        # Datos de salida
        rho = 903 - 0.672 * Tf
        Cf = 1820 + 3.478 * Tf
        # Kf = 0.1923 - 1.3e-4 * Tf
        # mu = 3.558 - 0.0435 * Tf + 2.2860e-4 * Tf ** 2 - 5.5037e-7 * Tf ** 3 + 4.9425e-10 * Tf ** 4
        # Pr = 212 - 2.2786 * Tf + 8.97e-3 * Tf ** 2 - 1.2e-5 * Tf ** 3
        # Hv = 2.17 * 1e6 - 5.01e4 * Tf + 4.53e2 * Tf ** 2 - 1.64 * Tf ** 3 + 2.1e-3 * Tf ** 4
        # Ht = Hv * (q / 1000) ** 0.8
        # Ent = 0.001765540717381 * Tf ** 2 + 1.839633987714124 * Tf + 31.665465197994973

        return rho, Cf


if __name__ == "__main__":
    """ Test del modelo. """

    # Definición del modelo
    model = ACUREXModel(integration_time=0.25)

    # Pruebas
    geometric_eff = model.calculate_geometric_efficiency(hour=17.54,
                                                         date={'day': 10, 'month': 10, 'year': 2005})
    Tout = model.predict(last_outlet_temp=221,
                         flow=0.68,
                         irradiance=893,
                         inlet_temp=126,
                         ambient_temp=31,
                         geometric_efficiency=geometric_eff)

    # Resultados
    print(geometric_eff)
    print(Tout)
