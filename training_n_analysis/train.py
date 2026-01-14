import gymnasium as gym
from sb3_contrib import MaskablePPO
import os
import numpy as np
import sys 

import monopoly_env 

from Single_Agent_Wrapper import SingleAgentMonopolyWrapper
from agents.Randy import Randy
from agents.Terry import Terry
from agents.Allin import Allin
from agents.Passive import Passive
from agents.PPO_Agent import PPO_Agent

# Configuraciones
MODELS_DIR = "models/ppo_league"
LOG_DIR = "logs/ppo_league"
# WIN_RATE_THRESHOLD = 0.75 
N_EVAL_EPISODES = 1000
TRAIN_STEPS_PER_CYCLE = 60000 

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# --- CLASES DE UTILIDAD ---
class DummyFile:
    def write(self, x): pass
    def flush(self): pass

class SilenceWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self._dummy = DummyFile()

    def step(self, action):
        original_stdout = sys.stdout
        sys.stdout = self._dummy 
        try:
            return self.env.step(action)
        finally:
            sys.stdout = original_stdout 

    def reset(self, **kwargs):
        original_stdout = sys.stdout
        sys.stdout = self._dummy
        try:
            return self.env.reset(**kwargs)
        finally:
            sys.stdout = original_stdout
    
    def action_masks(self):
        if hasattr(self.env, 'action_masks'):
            return self.env.action_masks()
        return self.env.unwrapped._get_action_mask()

class TrainingDummyPlayer:
    def __init__(self):
        self.name = "Learning_Hero"
        self.color = (0, 255, 0)
        self.order = 0

def make_env(opponents, render_mode=None, silent=True):
    players = [TrainingDummyPlayer()] + opponents
    
    env = gym.make(
        'MonopolyEnv-v0',
        players=players,
        render_mode=render_mode,
        max_steps=20000, 
        board_names_path='cards/f1_board_names.txt',
        community_chest_path='cards/f1_community_chest.txt',
        chance_path='cards/f1_chance.txt',
        hard_rules=False,
        image_path='cards/monopoly.png'
    )
    
    unwrapped_env = env.unwrapped
    current_players = list(unwrapped_env.players)
    current_players.sort(key=lambda p: 0 if p.name == "Learning_Hero" else 1)
    unwrapped_env.players = current_players
    
    for i, p in enumerate(unwrapped_env.players):
        p.order = i 
    
    unwrapped_env.star_order = lambda: None

    env = SingleAgentMonopolyWrapper(env, opponents)
    
    if silent:
        env = SilenceWrapper(env)
        
    return env

def evaluate_agent(model, opponents, n_games=50):
    print(f"\n   -> Evaluando... ({n_games} partidas)")
    eval_env = make_env(opponents, render_mode=None, silent=True)
    
    wins = 0
    for i in range(n_games):
        obs, info = eval_env.reset()
        done = False
        truncated = False
        
        while not (done or truncated):
            action_masks = eval_env.action_masks()
            action, _ = model.predict(obs, action_masks=action_masks, deterministic=True)
            obs, reward, done, truncated, info = eval_env.step(action)
            
            if done:
                winner_idx = eval_env.env.unwrapped._check_for_winner()
                if winner_idx == 0:
                    wins += 1
    
    eval_env.close()
    win_rate = wins / n_games
    print(f"   -> Resultado: {wins}/{n_games} victorias ({win_rate:.2%})")
    return win_rate

def run_training_loop():
    # 1. Definir rutas
    best_model_path = f"{MODELS_DIR}/best_model"
    best_model_zip = f"{best_model_path}.zip"
    
    generation = 0
    model = None
    
    # 2. Configurar oponentes iniciales (Heurísticos por defecto)
    # Estos se usan si se empieza de cero
    current_opponents = [
        Randy(), 
        Randy("Randy_Blue", (0,0,200)),
        Terry(),
        Terry("Terry_Green", (0,200,0)),
        Allin(),
        Passive()
    ]
    
    # Crear entorno temporal para inicializar modelo
    training_env = make_env(current_opponents, render_mode=None, silent=True)

    # 3. Lógica de Carga / Inicio
    if os.path.exists(best_model_zip):
        print(f"🔄 Modelo encontrado: {best_model_zip}")
        print("   -> Cargando pesos y configurando para Generación 2...")
        
        # Cargar modelo existente
        model = MaskablePPO.load(best_model_zip, env=training_env)
        generation = 2
        
        # CONFIGURACIÓN SELF-PLAY:
        # Como ya tenemos un modelo "bueno" (Gen 2), lo usamos como oponente
        # mezclado con heurísticos para mantener variedad.
        print("   -> Actualizando oponentes a Self-Play (Gen 2 + Heurísticos)...")
        current_opponents = [
            PPO_Agent(f"Gen_{generation}_A", (200,0,0), best_model_path),
            PPO_Agent(f"Gen_{generation}_B", (0,0,200), best_model_path),
            # Mantenemos algunos heurísticos para que no olvide cómo ganarles
            Terry("Terry_Sparring"),
            Allin("Allin_Sparring"),
            Randy("Randy_Sparring"),
            Passive("Passive_Sparring")
        ]
        
        # Reconstruir el entorno con los nuevos oponentes fuertes
        training_env.close()
        training_env = make_env(current_opponents, render_mode=None, silent=True)
        model.set_env(training_env)
        WIN_RATE_THRESHOLD = 0.45
        
    else:
        print("✨ No se encontró modelo previo. Iniciando entrenamiento desde CERO (Gen 0)...")
        # Iniciar modelo nuevo
        model = MaskablePPO(
            "MlpPolicy", 
            training_env, 
            verbose=1, 
            learning_rate=0.0003, 
            n_steps=2048, 
            batch_size=64,
            tensorboard_log=LOG_DIR,
            gamma=0.99
        )
        WIN_RATE_THRESHOLD = 0.75

    # 4. Bucle de Entrenamiento
    while True:
        print(f"\n--- Generación {generation}: Entrenando... ---")
        try:
            opp_names = [p.name for p in training_env.env.opponents]
        except AttributeError:
            opp_names = ["Oponentes Genéricos"]
        print(f"   VS: {opp_names}")
        
        # A. Entrenar
        # reset_num_timesteps=False permite acumular las gráficas en Tensorboard
        model.learn(total_timesteps=TRAIN_STEPS_PER_CYCLE, reset_num_timesteps=False, progress_bar=True)
        
        # B. Evaluar
        win_rate = evaluate_agent(model, current_opponents, n_games=N_EVAL_EPISODES)
        
        # C. Evolucionar Liga
        if win_rate >= WIN_RATE_THRESHOLD:
            generation += 1
            print(f"🚀 ¡NIVEL SUPERADO! ({win_rate:.1%} >= {WIN_RATE_THRESHOLD:.1%})")
            
            # Guardar nueva versión
            new_path = f"{MODELS_DIR}/gen_{generation}"
            model.save(new_path)
            model.save(best_model_path) # Sobrescribe el 'best_model' para la próxima vez
            
            print("   -> Actualizando oponentes con la nueva versión...")
            
            # Actualizar pool de oponentes:
            # 2 Clones de la nueva generación + Heurísticos
            current_opponents = [
                PPO_Agent(f"Gen_{generation}_A", (200,0,0), new_path),
                PPO_Agent(f"Gen_{generation}_B", (0,0,200), new_path),
                Terry(),
                Allin(),
                Randy(),
                Passive()
            ]
            
            training_env.close()
            training_env = make_env(current_opponents, render_mode=None, silent=True)
            model.set_env(training_env)
        else:
            print(f"🐢 Nivel no superado ({win_rate:.1%} < {WIN_RATE_THRESHOLD:.1%}). Entrenando más...")

if __name__ == '__main__':
    run_training_loop()