import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import torch
import torch.nn as nn
import torch.optim as optim
from deap import base, creator, tools
import matplotlib.pyplot as plt
import random
import time

g, l, h = 9.8, 0.3, 0.02

def simulate(F, theta, theta_dot, x, x_dot):
    sin_theta, cos_theta = np.sin(theta), np.cos(theta)
    theta_ddot = (g * sin_theta - F * cos_theta) / l
    x_ddot = F
    x_dot += h * x_ddot
    x += h * x_dot
    theta_dot += h * theta_ddot
    theta += h * theta_dot
    return x, x_dot, theta, theta_dot

def create_fis_system():
    angle = ctrl.Antecedent(np.arange(-1, 1.01, 0.01), 'angle')
    angle_dot = ctrl.Antecedent(np.arange(-5, 5.1, 0.1), 'angle_dot')
    force = ctrl.Consequent(np.arange(-100, 101, 1), 'force')

    angle['N'] = fuzz.trimf(angle.universe, [-1, -1, 0])
    angle['Z'] = fuzz.trimf(angle.universe, [-1, 0, 1])
    angle['P'] = fuzz.trimf(angle.universe, [0, 1, 1])

    angle_dot['N'] = fuzz.trimf(angle_dot.universe, [-5, -5, 0])
    angle_dot['Z'] = fuzz.trimf(angle_dot.universe, [-5, 0, 5])
    angle_dot['P'] = fuzz.trimf(angle_dot.universe, [0, 5, 5])

    force['NL'] = fuzz.trimf(force.universe, [-100, -100, -50])
    force['Z'] = fuzz.trimf(force.universe, [-50, 0, 50])
    force['PL'] = fuzz.trimf(force.universe, [50, 100, 100])

    rules = [
        ctrl.Rule(angle['N'] & angle_dot['N'], force['NL']),
        ctrl.Rule(angle['Z'] & angle_dot['Z'], force['Z']),
        ctrl.Rule(angle['P'] & angle_dot['P'], force['PL']),
    ]

    system = ctrl.ControlSystem(rules)
    return system

class NeuroFIS(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.net(x)

def train_neuro_fis():
    fis_system = create_fis_system()
    X, y = [], []
    for _ in range(500):
        a = random.uniform(-1, 1)
        ad = random.uniform(-3, 3)
        sim = ctrl.ControlSystemSimulation(fis_system)
        sim.input['angle'] = a
        sim.input['angle_dot'] = ad
        sim.compute()
        force = sim.output['force']
        X.append([a, ad, 0, 0])
        y.append(force)
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).view(-1,1)
    model = NeuroFIS()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()
    for _ in range(100):
        optimizer.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
    return model

def optimize_fis_with_genetic():
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)
    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.uniform, 0.5, 2.0)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, 1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def eval_individual(individual):
        scale = individual[0]
        fis_system = create_fis_system()
        x, x_dot, theta, theta_dot = 0, 0, 0.1, 0
        total_error = 0
        for _ in range(100):
            sim = ctrl.ControlSystemSimulation(fis_system)
            sim.input['angle'] = theta
            sim.input['angle_dot'] = theta_dot
            sim.compute()
            F = sim.output['force'] * scale
            x, x_dot, theta, theta_dot = simulate(F, theta, theta_dot, x, x_dot)
            total_error += abs(theta) + abs(x)
        return (total_error,)

    toolbox.register("evaluate", eval_individual)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=1.0, sigma=0.1, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    pop = toolbox.population(n=10)
    best_errors = []
    for _ in range(5):
        offspring = list(map(toolbox.clone, toolbox.select(pop, len(pop))))
        for c1, c2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < 0.5:
                toolbox.mate(c1, c2)
            if random.random() < 0.2:
                toolbox.mutate(c1)
        for ind in offspring:
            ind.fitness.values = toolbox.evaluate(ind)
        pop[:] = offspring

        best = tools.selBest(pop, 1)[0]
        best_errors.append(best.fitness.values[0])

    best = tools.selBest(pop, 1)[0]
    return create_fis_system(), best[0], best_errors

def run_simulation(controller, kind='fis', scale=1.0):
    x, x_dot, theta, theta_dot = 0, 0, 0.1, 0
    states, forces = [], []
    start_time = time.time()
    for _ in range(500):
        if kind in ['fis', 'genetico']:
            sim = ctrl.ControlSystemSimulation(controller)
            sim.input['angle'] = theta
            sim.input['angle_dot'] = theta_dot
            sim.compute()
            F = sim.output['force'] * scale
        elif kind == 'neuro':
            inp = torch.tensor([[theta, theta_dot, x, x_dot]], dtype=torch.float32)
            F = controller(inp).item()
        else:
            F = 0
        x, x_dot, theta, theta_dot = simulate(F, theta, theta_dot, x, x_dot)
        states.append([x, x_dot, theta, theta_dot])
        forces.append(F)
    elapsed = time.time() - start_time
    states = np.array(states)
    forces = np.array(forces)
    avg_error = np.mean(np.abs(states[:, 0]) + np.abs(states[:, 2]))
    var_error = np.var(np.abs(states[:, 0]) + np.abs(states[:, 2]))
    avg_time = elapsed / 500
    return states, forces, avg_error, var_error, avg_time

def compare_all():
    fis_system = create_fis_system()
    neuro_model = train_neuro_fis()
    genetico_system, scale_factor, gen_errors = optimize_fis_with_genetic()

    fis_states, fis_forces, fis_error, fis_var, fis_time = run_simulation(fis_system, 'fis')
    genetico_states, genetico_forces, genetico_error, genetico_var, genetico_time = run_simulation(genetico_system, 'genetico', scale_factor)
    neuro_states, neuro_forces, neuro_error, neuro_var, neuro_time = run_simulation(neuro_model, 'neuro')

    print(f"FIS: Erro médio = {fis_error:.4f}, Variância = {fis_var:.4f}, Tempo médio = {fis_time:.6f} s")
    print(f"Genético-FIS: Erro médio = {genetico_error:.4f}, Variância = {genetico_var:.4f}, Tempo médio = {genetico_time:.6f} s")
    print(f"Neuro-FIS: Erro médio = {neuro_error:.4f}, Variância = {neuro_var:.4f}, Tempo médio = {neuro_time:.6f} s")

    fig, axs = plt.subplots(4,1, figsize=(10,16), sharex=True)

    axs[0].plot(fis_forces, label=f'FIS (Erro={fis_error:.3f}, Var={fis_var:.3f}, Tempo={fis_time:.5f}s)')
    axs[0].plot(genetico_forces, label=f'Genético-FIS (Erro={genetico_error:.3f}, Var={genetico_var:.3f}, Tempo={genetico_time:.5f}s)')
    axs[0].plot(neuro_forces, label=f'Neuro-FIS (Erro={neuro_error:.3f}, Var={neuro_var:.3f}, Tempo={neuro_time:.5f}s)')
    axs[0].set_title('Força Aplicada')
    axs[0].legend()
    axs[0].grid()

    axs[1].plot(fis_states[:,2], label=f'FIS (Erro={fis_error:.3f}, Var={fis_var:.3f})')
    axs[1].plot(genetico_states[:,2], label=f'Genético-FIS (Erro={genetico_error:.3f}, Var={genetico_var:.3f})')
    axs[1].plot(neuro_states[:,2], label=f'Neuro-FIS (Erro={neuro_error:.3f}, Var={neuro_var:.3f})')
    axs[1].set_title('Ângulo do Pêndulo')
    axs[1].legend()
    axs[1].grid()

    axs[2].plot(fis_states[:,0], label=f'FIS (Erro={fis_error:.3f}, Var={fis_var:.3f})')
    axs[2].plot(genetico_states[:,0], label=f'Genético-FIS (Erro={genetico_error:.3f}, Var={genetico_var:.3f})')
    axs[2].plot(neuro_states[:,0], label=f'Neuro-FIS (Erro={neuro_error:.3f}, Var={neuro_var:.3f})')
    axs[2].set_title('Posição do Carrinho')
    axs[2].legend()
    axs[2].grid()

    axs[3].plot(gen_errors, marker='o')
    axs[3].set_title("Convergência do Algoritmo Genético")
    axs[3].set_xlabel("Geração")
    axs[3].set_ylabel("Melhor Erro (fitness)")
    axs[3].grid()

    plt.tight_layout()
    plt.savefig("comparacao_final_controles_completo.png")
    plt.show()

if __name__ == "__main__":
    print("Iniciando simulação:", time.ctime())
    compare_all()
    print("Simulação concluída:", time.ctime())
