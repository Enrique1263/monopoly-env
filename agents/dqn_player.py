import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from player import Player
import random
from collections import deque

# =========================
# Red neuronal mejorada con Dueling DQN
# =========================
class DuelingDQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DuelingDQN, self).__init__()
        
        # Feature extraction
        self.feature = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        features = self.feature(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Dueling formula: Q = V + (A - mean(A))
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values

# =========================
# Normalización mejorada del estado
# =========================
def flatten_state(game_state, num_players=4, state_dim=200):
    """
    Normaliza el estado para que la red aprenda mejor.
    """
    flat = []
    
    # PROPERTIES (0-39): -2, -1, 0-3 → normalizar a [-1, 1]
    properties = np.array(game_state[0], dtype=np.float32)
    properties_norm = properties / max(num_players - 1, 1)
    flat.extend(properties_norm)
    
    # BUILDINGS (40-79): -2 a 5 → normalizar a [-1, 1]
    buildings = np.array(game_state[5], dtype=np.float32)
    buildings_norm = (buildings + 2) / 7.0 - 1
    flat.extend(buildings_norm)
    
    # POSITIONS (80-83): 0 a 39 → normalizar a [0, 1]
    positions = np.array(game_state[1], dtype=np.float32)
    positions_norm = positions / 39.0
    flat.extend(positions_norm)
    
    # MONEY (84-87): 0 a ~5000 → normalizar con log
    money = np.array(game_state[2], dtype=np.float32)
    money_norm = np.log1p(money) / np.log1p(5000)
    flat.extend(money_norm)
    
    # JAIL_TURNS (88-91): 0 a 3 → normalizar a [0, 1]
    jail = np.array(game_state[6], dtype=np.float32)
    jail_norm = jail / 3.0
    flat.extend(jail_norm)
    
    # GET_OUT_OF_JAIL (92-95): 0 a 4 → normalizar a [0, 1]
    jail_cards = np.array(game_state[8], dtype=np.float32)
    jail_cards_norm = jail_cards / 4.0
    flat.extend(jail_cards_norm)
    
    # Características adicionales
    for player_id in range(num_players):
        num_props = np.sum(properties == player_id)
        flat.append(num_props / 28.0)
    
    for player_id in range(num_players):
        monopoly_count = 0
        groups = [[1,3], [6,8,9], [11,13,14], [16,18,19], 
                  [21,23,24], [26,27,29], [31,32,34], [37,39]]
        for group in groups:
            if all(properties[idx] == player_id for idx in group):
                monopoly_count += 1
        flat.append(monopoly_count / 8.0)
    
    # Rellenar o truncar
    if len(flat) < state_dim:
        flat += [0.0] * (state_dim - len(flat))
    elif len(flat) > state_dim:
        flat = flat[:state_dim]
    
    return np.array(flat, dtype=np.float32)

# =========================
# Prioritized Experience Replay
# =========================
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        
    def add(self, state, action, reward, next_state, done):
        max_priority = self.priorities.max() if self.buffer else 1.0
        max_priority = max(max_priority, 1e-6)
        
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)
        
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:len(self.buffer)]
        
        priorities = np.maximum(priorities, 1e-6)
        
        probs = priorities ** self.alpha
        probs_sum = probs.sum()
        
        if probs_sum <= 0 or np.isnan(probs_sum) or np.isinf(probs_sum):
            probs = np.ones(len(self.buffer)) / len(self.buffer)
        else:
            probs /= probs_sum
        
        probs = np.nan_to_num(probs, nan=1.0/len(self.buffer), posinf=1.0/len(self.buffer))
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=False)
        samples = [self.buffer[idx] for idx in indices]
        
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights = np.nan_to_num(weights, nan=1.0, posinf=1.0)
        
        if weights.max() > 0:
            weights /= weights.max()
        else:
            weights = np.ones_like(weights)
        
        return samples, indices, weights
    
    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = max(float(priority), 1e-6)
    
    def __len__(self):
        return len(self.buffer)

# =========================
# Jugador DQN Optimizado Anti-Olvido
# =========================
class DQNPlayer(Player):
    def __init__(self, name='DQN', color=(0,255,0), state_dim=200, action_dim=43, num_players=4):
        """
        ⚠️ IMPORTANTE: Configurado para prevenir olvido catastrófico
        """
        super().__init__(name, color)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_players = num_players
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Red neuronal con Dueling DQN
        self.policy_net = DuelingDQN(state_dim, action_dim).to(self.device)
        self.target_net = DuelingDQN(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Prioritized Replay buffer
        self.memory = PrioritizedReplayBuffer(capacity=50000, alpha=0.6)
        
        # Optimizador con learning rate MÁS BAJO para prevenir olvido
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=5e-5)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=2000, gamma=0.95)

        # Hiperparámetros OPTIMIZADOS ANTI-OLVIDO
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.05  # MÁS ALTO: mantiene exploración continua
        self.epsilon_decay = 0.9998  # MÁS LENTO: decay gradual
        self.batch_size = 128
        self.beta = 0.4
        self.beta_increment = 0.0005  # MÁS LENTO: menos agresivo
        
        # Contadores
        self.steps = 0
        self.update_target_every = 2000  # MÁS ESPACIADO: estabilidad

    def action(self, observation, valid_mask=None):
        """
        Selecciona acción usando epsilon-greedy con manejo robusto de valid_mask
        """
        state = flatten_state(observation, num_players=self.num_players, state_dim=self.state_dim)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        # Epsilon-greedy
        if random.random() < self.epsilon:
            if valid_mask is not None:
                valid_actions = np.where(valid_mask)[0]
                if len(valid_actions) == 0:
                    return 0
                return int(np.random.choice(valid_actions))
            else:
                return random.randint(0, self.action_dim-1)
        else:
            with torch.no_grad():
                q_values = self.policy_net(state_tensor).cpu().numpy().flatten()
                
                if valid_mask is not None:
                    if len(valid_mask) < self.action_dim:
                        valid_mask = np.pad(
                            valid_mask, 
                            (0, self.action_dim - len(valid_mask)), 
                            constant_values=False
                        )
                    elif len(valid_mask) > self.action_dim:
                        valid_mask = valid_mask[:self.action_dim]
                    
                    masked_q = np.where(valid_mask, q_values, -np.inf)
                    
                    if np.all(np.isinf(masked_q)):
                        valid_actions = np.where(valid_mask)[0]
                        return int(valid_actions[0]) if len(valid_actions) > 0 else 0
                    
                    return int(np.argmax(masked_q))
                
                return int(np.argmax(q_values))

    def remember(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

    def optimize(self):
        """
        Optimización con protecciones anti-olvido
        """
        if len(self.memory) < self.batch_size:
            return 0.0
        
        # Sample con prioridades
        batch, indices, weights = self.memory.sample(self.batch_size, self.beta)
        self.beta = min(0.8, self.beta + self.beta_increment)  # Limitado a 0.8
        
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        weights = torch.FloatTensor(weights).unsqueeze(1).to(self.device)

        # Double DQN
        q_values = self.policy_net(states).gather(1, actions)
        
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(1, keepdim=True)
            q_next = self.target_net(next_states).gather(1, next_actions)
            q_target = rewards + self.gamma * q_next * (1 - dones)
            q_target = torch.clamp(q_target, -1000, 1000)

        # Huber loss
        td_errors = q_target - q_values
        loss = (weights * td_errors.pow(2)).mean()
        
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"⚠️ WARNING: Loss anómalo: {loss.item()}")
            return 0.0
        
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping conservador
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
        
        self.optimizer.step()

        # Actualizar prioridades
        priorities = td_errors.abs().detach().cpu().numpy() + 1e-6
        self.memory.update_priorities(indices, priorities.flatten())

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Actualizar target network
        self.steps += 1
        if self.steps % self.update_target_every == 0:
            self.update_target()
        
        return loss.item()

    def update_target(self):
        """Copia pesos de policy_net a target_net"""
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def save(self, path):
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps,
            'beta': self.beta
        }, path)
    
    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.steps = checkpoint['steps']
        if 'beta' in checkpoint:
            self.beta = checkpoint['beta']