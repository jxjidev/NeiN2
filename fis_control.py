# pendulum_control.py

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms
import random
import tensorflow as tf
from tensorflow.keras import layers, models
import time

# Parâmetros físicos
m_c = 0.5  # Massa do carrinho (kg)
m_p = 0.2  # Massa do pêndulo (kg)
l = 0.3    # Comprimento do pêndulo (m)
g = 9.8    # Gravidade (m/s^2)
I = 0.006  # Momento de inércia (kg.m^2)
h = 0.01   # Passo de tempo aumentado para acelerar (de 0.005)
b = 0.2    # Coeficiente de amortecimento

# Variáveis fuzzy
theta = ctrl.Antecedent(np.arange(-90, 91, 1), 'theta')  # Ângulo (graus)
dtheta = ctrl.Antecedent(np.arange(-200, 201, 1), 'dtheta')  # Velocidade angular (graus/s)
x = ctrl.Antecedent(np.arange(-10, 11, 0.1), 'x')  # Posição do carrinho
dx = ctrl.Antecedent(np.arange(-10, 11, 0.1), 'dx')  # Velocidade do carrinho
force = ctrl.Consequent(np.arange(-200, 201, 1), 'force')  # Força (N)

# Funções de pertinência
theta['N'] = fuzz.trapmf(theta.universe, [-90, -90, -45, 0])
theta['Z'] = fuzz.trimf(theta.universe, [-5, 0, 5])
theta['P'] = fuzz.trapmf(theta.universe, [0, 45, 90, 90])
dtheta['N'] = fuzz.trapmf(dtheta.universe, [-200, -200, -100, 0])
dtheta['Z'] = fuzz.trimf(dtheta.universe, [-20, 0, 20])
dtheta['P'] = fuzz.trapmf(dtheta.universe, [0, 100, 200, 200])
x['N'] = fuzz.trapmf(x.universe, [-10, -10, -5, 0])
x['Z'] = fuzz.trimf(x.universe, [-1, 0, 1])
x['P'] = fuzz.trapmf(x.universe, [0, 5, 10, 10])
dx['N'] = fuzz.trapmf(dx.universe, [-10, -10, -5, 0])
dx['Z'] = fuzz.trimf(dx.universe, [-1, 0, 1])
dx['P'] = fuzz.trapmf(dx.universe, [0, 5, 10, 10])
force['NL'] = fuzz.trapmf(force.universe, [-200, -200, -120, -60])
force['NM'] = fuzz.trimf(force.universe, [-120, -60, 0])
force['NS'] = fuzz.trimf(force.universe, [-60, -20, 0])
force['Z'] = fuzz.trimf(force.universe, [-20, 0, 20])
force['PS'] = fuzz.trimf(force.universe, [0, 20, 60])
force['PM'] = fuzz.trimf(force.universe, [0, 60, 120])
force['PL'] = fuzz.trapmf(force.universe, [60, 120, 200, 200])

# Regras FIS
fis_rules = [
    ctrl.Rule(theta['N'] & dtheta['N'], force['PL']),
    ctrl.Rule(theta['N'] & dtheta['Z'], force['PM']),
    ctrl.Rule(theta['N'] & dtheta['P'], force['PS']),
    ctrl.Rule(theta['Z'] & dtheta['N'], force['PS']),
    ctrl.Rule(theta['Z'] & dtheta['Z'], force['Z']),
    ctrl.Rule(theta['Z'] & dtheta['P'], force['NS']),
    ctrl.Rule(theta['P'] & dtheta['N'], force['NS']),
    ctrl.Rule(theta['P'] & dtheta['Z'], force['NM']),
    ctrl.Rule(theta['P'] & dtheta['P'], force['NL']),
    ctrl.Rule(x['N'] & dx['N'], force['PL']),
    ctrl.Rule(x['N'] & dx['Z'], force['PM']),
    ctrl.Rule(x['N'] & dx['P'], force['PS']),
    ctrl.Rule(x['Z'] & dx['N'], force['PS']),
    ctrl.Rule(x['Z'] & dx['Z'], force['Z']),
    ctrl.Rule(x['Z'] & dx['P'], force['NS']),
    ctrl.Rule(x['P'] & dx['N'], force['NS']),
    ctrl.Rule(x['P'] & dx['Z'], force['NM']),
    ctrl.Rule(x['P'] & dx['P'], force['NL']),
]
fis_system = ctrl.ControlSystem(fis_rules)
fis_controller = ctrl.ControlSystemSimulation(fis_system)

# Funções dinâmicas
def derivatives(state, F):
    x, dx, theta, dtheta = state
    theta_rad = np.radians(theta)
    dtheta_rad = np.radians(dtheta)
    cos_theta = np.cos(theta_rad)
    sin_theta = np.sin(theta_rad)

    A = m_c + m_p
    B = m_p * l * cos_theta
    D = I + m_p * l**2
    E = F + m_p * l * dtheta_rad**2 * sin_theta - b * dx
    G = m_p * g * l * sin_theta - b * dtheta_rad

    det = A * D - B**2
    if det != 0:
        x_ddot = (D * E - B * G) / det
        theta_ddot = (-B * E + A * G) / det
    else:
        x_ddot = theta_ddot = 0
    return np.array([dx, x_ddot, dtheta, theta_ddot])

def rk4_step(state, F, h):
    k1 = derivatives(state, F)
    k2 = derivatives(state + 0.5 * h * k1, F)
    k3 = derivatives(state + 0.5 * h * k2, F)
    k4 = derivatives(state + h * k3, F)
    return state + (h / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

# Simulação FIS
t = np.arange(0, 5, h)
theta_vals_fis = np.zeros(len(t))
dtheta_vals_fis = np.zeros(len(t))
x_vals_fis = np.zeros(len(t))
dx_vals_fis = np.zeros(len(t))
force_vals_fis = np.zeros(len(t))

state = np.array([0, 0, 10, 0])
theta_vals_fis[0], dtheta_vals_fis[0], x_vals_fis[0], dx_vals_fis[0] = 10, 0, 0, 0

start_time = time.time()
for i in range(1, len(t)):
    x_vals_fis[i-1], dx_vals_fis[i-1], theta_vals_fis[i-1], dtheta_vals_fis[i-1] = state
    fis_controller.input['theta'] = theta_vals_fis[i-1]
    fis_controller.input['dtheta'] = dtheta_vals_fis[i-1]
    fis_controller.input['x'] = x_vals_fis[i-1]
    fis_controller.input['dx'] = dx_vals_fis[i-1]
    fis_controller.compute()
    F = fis_controller.output['force']
    force_vals_fis[i] = F
    state = rk4_step(state, F, h)
print(f"Tempo FIS: {time.time() - start_time:.2f}s")
x_vals_fis[-1], dx_vals_fis[-1], theta_vals_fis[-1], dtheta_vals_fis[-1] = state

# Configuração Genético-Fuzzy
if not hasattr(creator, "FitnessMin"):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMin)

def evaluate_rules(individual):
    temp_rules = []
    force_levels = ['NL', 'NM', 'NS', 'Z', 'PS', 'PM', 'PL']
    idx = 0
    for theta_state in ['N', 'Z', 'P']:
        for dtheta_state in ['N', 'Z', 'P']:
            temp_rules.append(ctrl.Rule(theta[theta_state] & dtheta[dtheta_state], force[force_levels[individual[idx]]]))
            idx += 1
    for x_state in ['N', 'Z', 'P']:
        for dx_state in ['N', 'Z', 'P']:
            temp_rules.append(ctrl.Rule(x[x_state] & dx[dx_state], force[force_levels[individual[idx]]]))
            idx += 1

    temp_system = ctrl.ControlSystem(temp_rules)
    temp_controller = ctrl.ControlSystemSimulation(temp_system)

    t_eval = np.arange(0, 1, h)  # Reduzido para 1 segundo
    state = np.array([0, 0, 10, 0])
    theta_sim = np.zeros(len(t_eval))
    dtheta_sim = np.zeros(len(t_eval))
    x_sim = np.zeros(len(t_eval))
    dx_sim = np.zeros(len(t_eval))
    theta_sim[0], dtheta_sim[0], x_sim[0], dx_sim[0] = 10, 0, 0, 0

    for i in range(1, len(t_eval)):
        x_sim[i-1], dx_sim[i-1], theta_sim[i-1], dtheta_sim[i-1] = state
        temp_controller.input['theta'] = theta_sim[i-1]
        temp_controller.input['dtheta'] = dtheta_sim[i-1]
        temp_controller.input['x'] = x_sim[i-1]
        temp_controller.input['dx'] = dx_sim[i-1]
        temp_controller.compute()
        F = temp_controller.output['force']
        state = rk4_step(state, F, h)
    x_sim[-1], dx_sim[-1], theta_sim[-1], dtheta_sim[-1] = state

    theta_error = np.mean(np.abs(theta_sim))
    x_error = np.mean(np.abs(x_sim))
    return (theta_error + x_error,)

toolbox = base.Toolbox()
toolbox.register("attr_int", random.randint, 0, 6)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, n=18)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate_rules)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutUniformInt, low=0, up=6, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

# Otimização Genético-Fuzzy
population = toolbox.population(n=10)  # Reduzido para 10
ngen = 5  # Reduzido para 5
start_time = time.time()
for gen in range(ngen):
    print(f"Geração {gen + 1}/{ngen}")
    offspring = algorithms.varAnd(population, toolbox, cxpb=0.7, mutpb=0.3)
    fits = toolbox.map(toolbox.evaluate, offspring)
    for fit, ind in zip(fits, offspring):
        ind.fitness.values = fit
    population = toolbox.select(offspring, k=len(population))
print(f"Tempo Genético-Fuzzy: {time.time() - start_time:.2f}s")

best_ind = tools.selBest(population, k=1)[0]
force_levels = ['NL', 'NM', 'NS', 'Z', 'PS', 'PM', 'PL']
optimized_rules = []
idx = 0
for theta_state in ['N', 'Z', 'P']:
    for dtheta_state in ['N', 'Z', 'P']:
        optimized_rules.append(ctrl.Rule(theta[theta_state] & dtheta[dtheta_state], force[force_levels[best_ind[idx]]]))
        idx += 1
for x_state in ['N', 'Z', 'P']:
    for dx_state in ['N', 'Z', 'P']:
        optimized_rules.append(ctrl.Rule(x[x_state] & dx[dx_state], force[force_levels[best_ind[idx]]]))
        idx += 1

gf_system = ctrl.ControlSystem(optimized_rules)
gf_controller = ctrl.ControlSystemSimulation(gf_system)

# Simulação Genético-Fuzzy
theta_vals_gf = np.zeros(len(t))
dtheta_vals_gf = np.zeros(len(t))
x_vals_gf = np.zeros(len(t))
dx_vals_gf = np.zeros(len(t))
force_vals_gf = np.zeros(len(t))

state = np.array([0, 0, 10, 0])
theta_vals_gf[0], dtheta_vals_gf[0], x_vals_gf[0], dx_vals_gf[0] = 10, 0, 0, 0

start_time = time.time()
for i in range(1, len(t)):
    x_vals_gf[i-1], dx_vals_gf[i-1], theta_vals_gf[i-1], dtheta_vals_gf[i-1] = state
    gf_controller.input['theta'] = theta_vals_gf[i-1]
    gf_controller.input['dtheta'] = dtheta_vals_gf[i-1]
    gf_controller.input['x'] = x_vals_gf[i-1]
    gf_controller.input['dx'] = dx_vals_gf[i-1]
    gf_controller.compute()
    F = gf_controller.output['force']
    force_vals_gf[i] = F
    state = rk4_step(state, F, h)
print(f"Tempo Simulação GF: {time.time() - start_time:.2f}s")
x_vals_gf[-1], dx_vals_gf[-1], theta_vals_gf[-1], dtheta_vals_gf[-1] = state

# Neuro-Fuzzy
inputs = []
outputs = []
for theta_val in np.linspace(-90, 90, 10):  # Reduzido de 20 para 10
    for dtheta_val in np.linspace(-200, 200, 10):  # Reduzido de 20 para 10
        for x_val in np.linspace(-10, 10, 5):  # Reduzido de 10 para 5
            for dx_val in np.linspace(-10, 10, 5):  # Reduzido de 10 para 5
                fis_controller.input['theta'] = theta_val
                fis_controller.input['dtheta'] = dtheta_val
                fis_controller.input['x'] = x_val
                fis_controller.input['dx'] = dx_val
                fis_controller.compute()
                inputs.append([theta_val, dtheta_val, x_val, dx_val])
                outputs.append(fis_controller.output['force'])

inputs = np.array(inputs)
outputs = np.array(outputs)

model = models.Sequential([
    layers.Dense(64, activation='relu', input_shape=(4,)),
    layers.Dropout(0.2),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse')
start_time = time.time()
model.fit(inputs, outputs, epochs=10, batch_size=32, validation_split=0.2, verbose=1)
print(f"Tempo Treinamento NF: {time.time() - start_time:.2f}s")

# Simulação Neuro-Fuzzy
theta_vals_nf = np.zeros(len(t))
dtheta_vals_nf = np.zeros(len(t))
x_vals_nf = np.zeros(len(t))
dx_vals_nf = np.zeros(len(t))
force_vals_nf = np.zeros(len(t))

state = np.array([0, 0, 10, 0])
theta_vals_nf[0], dtheta_vals_nf[0], x_vals_nf[0], dx_vals_nf[0] = 10, 0, 0, 0

start_time = time.time()
for i in range(1, len(t)):
    x_vals_nf[i-1], dx_vals_nf[i-1], theta_vals_nf[i-1], dtheta_vals_nf[i-1] = state
    input_data = np.array([[theta_vals_nf[i-1], dtheta_vals_nf[i-1], x_vals_nf[i-1], dx_vals_nf[i-1]]])
    F = model.predict(input_data, verbose=0)[0][0]
    force_vals_nf[i] = F
    state = rk4_step(state, F, h)
print(f"Tempo Simulação NF: {time.time() - start_time:.2f}s")
x_vals_nf[-1], dx_vals_nf[-1], theta_vals_nf[-1], dtheta_vals_nf[-1] = state

# Comparação
def calculate_metrics(theta_vals, x_vals):
    theta_mse = np.mean(theta_vals**2)
    x_mse = np.mean(x_vals**2)
    settling_time = next((i * h for i, val in enumerate(np.abs(theta_vals)) if val < 1), 5)
    return theta_mse, x_mse, settling_time

fis_metrics = calculate_metrics(theta_vals_fis, x_vals_fis)
gf_metrics = calculate_metrics(theta_vals_gf, x_vals_gf)
nf_metrics = calculate_metrics(theta_vals_nf, x_vals_nf)

print("Métricas de Desempenho (em 5s):")
print(f"FIS - MSE Ângulo: {fis_metrics[0]:.2f}, MSE Posição: {fis_metrics[1]:.2f}, Tempo de Estabilização: {fis_metrics[2]:.2f}s")
print(f"Genético-Fuzzy - MSE Ângulo: {gf_metrics[0]:.2f}, MSE Posição: {gf_metrics[1]:.2f}, Tempo de Estabilização: {gf_metrics[2]:.2f}s")
print(f"Neuro-Fuzzy - MSE Ângulo: {nf_metrics[0]:.2f}, MSE Posição: {nf_metrics[1]:.2f}, Tempo de Estabilização: {nf_metrics[2]:.2f}s")

# Gráficos
plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
plt.plot(t, theta_vals_fis, label='FIS')
plt.plot(t, theta_vals_gf, label='Genético-Fuzzy')
plt.plot(t, theta_vals_nf, label='Neuro-Fuzzy')
plt.grid()
plt.legend()
plt.xlabel('Tempo (s)')
plt.ylabel('Ângulo (graus)')
plt.title('Comparação do Ângulo')

plt.subplot(3, 1, 2)
plt.plot(t, x_vals_fis, label='FIS')
plt.plot(t, x_vals_gf, label='Genético-Fuzzy')
plt.plot(t, x_vals_nf, label='Neuro-Fuzzy')
plt.grid()
plt.legend()
plt.xlabel('Tempo (s)')
plt.ylabel('Posição')
plt.title('Comparação da Posição')

plt.subplot(3, 1, 3)
plt.plot(t, force_vals_fis, label='FIS')
plt.plot(t, force_vals_gf, label='Genético-Fuzzy')
plt.plot(t, force_vals_nf, label='Neuro-Fuzzy')
plt.grid()
plt.legend()
plt.xlabel('Tempo (s)')
plt.ylabel('Força (N)')
plt.title('Comparação da Força')

plt.tight_layout()
plt.show()