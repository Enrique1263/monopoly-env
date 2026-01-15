# train_dqn_monopoly.py
import os
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
import random
from agents.dqn_player import DQNPlayer, flatten_state
from agents.Randy import Randy
import gymnasium as gym
import monopoly_env

# =========================
# Configuración
# =========================
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)

NUM_EPISODES = 10000
SAVE_EVERY = 100
WINDOW = 100
STATE_DIM = 200
ACTION_DIM = 43
NUM_PLAYERS = 4

# PARÁMETROS OPTIMIZADOS para prevenir olvido catastrófico
TRAINING_STEPS_PER_EPISODE = 3  # Menos agresivo
TARGET_UPDATE_FREQ = 50  # Más estable

os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# Abrir log
log_file = open("logs/training_log.txt", "w")

def log_only(msg):
    log_file.write(msg + "\n")
    log_file.flush()

# =========================
# Crear agentes
# =========================
dqn1 = DQNPlayer(
    name="DQN_Hero", 
    state_dim=STATE_DIM, 
    action_dim=ACTION_DIM,
    num_players=NUM_PLAYERS, 
    color=(0, 255, 0)
)
bot1 = Randy(name="Randy1")
bot2 = Randy(name="Randy2")
bot3 = Randy(name="Randy3")

players = [dqn1, bot1, bot2, bot3]

for i, player in enumerate(players):
    player.order = i

# =========================
# Crear entorno
# =========================
env = gym.make('MonopolyEnv-v0',
               players=players,
               render_mode=None,
               max_steps=1200)

# =========================
# Métricas
# =========================
episode_rewards = []
episode_losses = []
win_counts_total = {player.name: 0 for player in players}
victories_window = {player.name: [] for player in players}
average_rewards = []
average_losses = []
dqn_win_rate = []
epsilon_history = []
q_values_history = []

# Métricas adicionales
truncated_count = 0
terminated_count = 0
truncated_window = []
terminated_window = []

best_win_rate = 0.0
patience = 0
max_patience = 50

# =========================
# Función de reward shaping
# =========================
def compute_shaped_reward(base_reward, current_state, next_state, done):
    shaped = base_reward
    
    if done and base_reward < -1000:
        shaped -= 50
    
    if done and base_reward > 1000:
        shaped += 50
    
    try:
        current_money = current_state[2][0]
        next_money = next_state[2][0]
        
        money_change = next_money - current_money
        shaped += money_change * 0.01
        
        if next_money > 500:
            shaped += 0.5
        
        if next_money < 100:
            shaped -= 1
    except:
        pass
    
    return shaped

# =========================
# Entrenamiento
# =========================
print("=" * 80)
print("🎮 ENTRENAMIENTO DQN - MONOPOLY (ANTI-OLVIDO)")
print("=" * 80)
print(f"Episodios: {NUM_EPISODES}")
print(f"ACTION_DIM: {ACTION_DIM}")
print(f"Training steps por episodio: {TRAINING_STEPS_PER_EPISODE}")
print(f"Target update cada: {TARGET_UPDATE_FREQ} episodios")
print("=" * 80)
print("🛡️ PROTECCIONES ANTI-OLVIDO:")
print("  ✅ Epsilon decay MÁS LENTO (0.9998)")
print("  ✅ Epsilon mínimo MÁS ALTO (0.05)")
print("  ✅ Learning rate MÁS BAJO (5e-5)")
print("  ✅ Menos optimizaciones/episodio (3)")
print("  ✅ Target update MÁS ESPACIADO (cada 50)")
print("  ✅ Detección de tendencias (⬆️⬇️➡️)")
print("=" * 80)

log_only("=" * 80)
log_only(f"ENTRENAMIENTO ANTI-OLVIDO - ACTION_DIM={ACTION_DIM}")
log_only("=" * 80)

start_time = time.time()

for episode in range(1, NUM_EPISODES + 1):
    obs, info = env.reset()
    terminated = False
    truncated = False
    done = False
    total_reward = 0
    episode_loss = []
    episode_q_values = []
    
    transitions = []

    while not done:
        current_idx = info["player_on_turn"]
        current_state = info["game_state"]
        player = players[current_idx]
        valid_mask = info.get("action_mask", None)

        if isinstance(player, DQNPlayer):
            action = player.action(current_state, valid_mask=valid_mask)
            
            with torch.no_grad():
                state_tensor = torch.FloatTensor(
                    flatten_state(current_state, NUM_PLAYERS, STATE_DIM)
                ).unsqueeze(0).to(player.device)
                q_vals = player.policy_net(state_tensor)
                episode_q_values.append(q_vals.max().item())
        else:
            action = player.action(current_state)

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        if isinstance(player, DQNPlayer):
            s = flatten_state(current_state, NUM_PLAYERS, STATE_DIM)
            ns = flatten_state(info["game_state"], NUM_PLAYERS, STATE_DIM)
            
            shaped_reward = compute_shaped_reward(reward, current_state, 
                                                  info["game_state"], done)
            
            transitions.append((s, action, shaped_reward, ns, done))
            total_reward += reward

        obs = next_obs
    
    # Entrenamiento
    if isinstance(players[0], DQNPlayer):
        for s, a, r, ns, d in transitions:
            dqn1.remember(s, a, r, ns, d)
        
        if len(dqn1.memory) >= dqn1.batch_size:
            for _ in range(TRAINING_STEPS_PER_EPISODE):
                loss = dqn1.optimize()
                if loss > 0:
                    episode_loss.append(loss)
    
    # Actualizar target network
    if episode % TARGET_UPDATE_FREQ == 0:
        dqn1.update_target()
    
    # Métricas
    episode_rewards.append(total_reward)
    if episode_loss:
        episode_losses.append(np.mean(episode_loss))
    else:
        episode_losses.append(0)
    
    epsilon_history.append(dqn1.epsilon)
    
    if episode_q_values:
        q_values_history.append(np.mean(episode_q_values))
    else:
        q_values_history.append(0)

    # Contador de truncated vs terminated
    if truncated:
        truncated_count += 1
    if terminated:
        terminated_count += 1

    # Ganador detectado correctamente
    winner_name = "Nobody"
    if terminated:
        try:
            bankrupt = info["game_state"][7]
            players_alive = [i for i, is_bankrupt in enumerate(bankrupt) if not is_bankrupt]
            
            if len(players_alive) == 1:
                winner_idx = players_alive[0]
                winner_name = players[winner_idx].name
                win_counts_total[winner_name] += 1
            else:
                money_list = info["game_state"][2]
                winner_idx = int(np.argmax(money_list))
                winner_name = players[winner_idx].name + " (by_money)"
                win_counts_total[winner_name] += 1
        except Exception as e:
            print(f"⚠️ Error determinando ganador ep {episode}: {e}")
            winner_name = "Unknown"
    elif truncated:
        winner_name = "Truncated"

    # Logging cada WINDOW
    if episode % WINDOW == 0:
        avg_reward = np.mean(episode_rewards[-WINDOW:])
        avg_loss = np.mean(episode_losses[-WINDOW:])
        avg_q = np.mean(q_values_history[-WINDOW:]) if q_values_history else 0
        average_rewards.append(avg_reward)
        average_losses.append(avg_loss)
        
        # Win rate sobre partidas terminadas
        total_real_wins = sum(win_counts_total.values())
        dqn_wins = win_counts_total[dqn1.name]
        
        if terminated_count > 0:
            current_win_rate = (dqn_wins / terminated_count) * 100
        else:
            current_win_rate = 0.0
        
        dqn_win_rate.append(current_win_rate)
        
        # Victorias en ventana
        if episode >= WINDOW:
            recent_dqn_wins = dqn_wins - (
                victories_window[dqn1.name][-1] if victories_window[dqn1.name] 
                else 0
            )
            recent_truncated = truncated_count - (truncated_window[-1] if truncated_window else 0)
            recent_terminated = terminated_count - (terminated_window[-1] if terminated_window else 0)
        else:
            recent_dqn_wins = dqn_wins
            recent_truncated = truncated_count
            recent_terminated = terminated_count
        
        truncated_window.append(truncated_count)
        terminated_window.append(terminated_count)
        
        if recent_terminated > 0:
            recent_win_rate = (recent_dqn_wins / recent_terminated) * 100
        else:
            recent_win_rate = 0.0
        
        for name in win_counts_total:
            victories_window[name].append(win_counts_total[name])
        
        # Tendencia de win rate
        trend_symbol = "➡️"
        win_rate_change = 0
        if len(dqn_win_rate) > 1:
            win_rate_change = dqn_win_rate[-1] - dqn_win_rate[-2]
            if win_rate_change > 1:
                trend_symbol = "⬆️"
            elif win_rate_change < -1:
                trend_symbol = "⬇️"
        
        # Warnings
        collapse_warning = ""
        if avg_loss < 0.0001 and episode > 500:
            collapse_warning = " ⚠️ Loss muy bajo"
        if avg_q < 1 and episode > 500:
            collapse_warning += " ⚠️ Q-values bajos"
        if win_rate_change < -5 and episode > 500:
            collapse_warning += " 🚨 OLVIDO DETECTADO"
        
        # PRINT
        print(f"\n{'='*80}")
        print(f"📈 Ep {episode}/{NUM_EPISODES}{collapse_warning}")
        print(f"{'-'*80}")
        print(f"Últimos {WINDOW} eps:")
        print(f"  • Wins DQN: {recent_dqn_wins}/{recent_terminated} ({recent_win_rate:.1f}%)")
        print(f"  • Terminadas: {recent_terminated} | Truncadas: {recent_truncated}")
        print(f"  • Avg Reward: {avg_reward:+.2f}")
        print(f"  • Avg Loss: {avg_loss:.4f}")
        print(f"  • Avg Q: {avg_q:.2f}")
        print(f"{'-'*80}")
        print(f"Total Acumulado:")
        print(f"  • DQN: {dqn_wins}/{terminated_count} ({current_win_rate:.2f}%) {trend_symbol} {win_rate_change:+.1f}%")
        print(f"  • Terminadas: {terminated_count}/{episode} ({(terminated_count/episode)*100:.1f}%)")
        print(f"  • Truncadas: {truncated_count}/{episode} ({(truncated_count/episode)*100:.1f}%)")
        print(f"{'-'*80}")
        print(f"Estado:")
        print(f"  • ε: {dqn1.epsilon:.4f}")
        print(f"  • Memoria: {len(dqn1.memory)}/{dqn1.memory.capacity}")
        print(f"  • LR: {dqn1.optimizer.param_groups[0]['lr']:.6f}")
        print(f"{'='*80}\n")
        
        # Log
        log_only(f"\nEpisodio {episode}/{NUM_EPISODES}")
        log_only(f"Win Rate: {current_win_rate:.2f}% | Recent: {recent_win_rate:.1f}% | Trend: {win_rate_change:+.1f}%")
        log_only(f"Terminadas: {terminated_count} | Truncadas: {truncated_count}")
        log_only(f"Reward: {avg_reward:+.2f} | Loss: {avg_loss:.4f} | Q: {avg_q:.2f}")
        log_only(f"Epsilon: {dqn1.epsilon:.4f}")
        
        # Early stopping mejorado
        if current_win_rate > best_win_rate:
            best_win_rate = current_win_rate
            patience = 0
            dqn1.save(f"models/best_model.pt")
            print(f"💾 Mejor modelo guardado (WR: {best_win_rate:.2f}%)\n")
            log_only(f"Nuevo mejor modelo: {best_win_rate:.2f}%")
        else:
            patience += 1
        
        if patience >= max_patience and episode > 2000:
            print(f"\n⚠️  Early stopping: Sin mejora en {patience} ventanas")
            log_only(f"Early stopping en episodio {episode}")
            break

    # Checkpoints
    if episode % SAVE_EVERY == 0:
        dqn1.save(f"models/dqn_episode{episode}.pt")
        log_only(f"Checkpoint: episode{episode}.pt")

# =========================
# Fin
# =========================
end_time = time.time()
total_time = end_time - start_time

print("\n" + "="*80)
print("🏁 ENTRENAMIENTO COMPLETADO")
print(f"⏱️  Tiempo: {total_time/60:.2f} min")
if terminated_count > 0:
    print(f"🏆 Win Rate Final: {(win_counts_total[dqn1.name]/terminated_count)*100:.2f}%")
else:
    print(f"🏆 Win Rate Final: N/A (no hubo partidas terminadas)")
print(f"🥇 Mejor: {best_win_rate:.2f}%")
print(f"📊 Terminadas: {terminated_count}/{episode} ({(terminated_count/episode)*100:.1f}%)")
print(f"⏱️  Truncadas: {truncated_count}/{episode} ({(truncated_count/episode)*100:.1f}%)")
print("="*80)

log_only("\n" + "="*80)
log_only("ENTRENAMIENTO COMPLETADO")
log_only(f"Tiempo: {total_time/60:.2f} min")
if terminated_count > 0:
    log_only(f"Win Rate Final: {(win_counts_total[dqn1.name]/terminated_count)*100:.2f}%")
log_only(f"Mejor: {best_win_rate:.2f}%")
log_only(f"Terminadas: {terminated_count} | Truncadas: {truncated_count}")

print("\n📊 RESULTADOS FINALES:")
for name, wins in win_counts_total.items():
    if terminated_count > 0:
        win_percentage = (wins / terminated_count) * 100
    else:
        win_percentage = 0
    print(f"   {name}: {wins}/{terminated_count} victorias ({win_percentage:.2f}%)")
    log_only(f"{name}: {wins} ({win_percentage:.2f}%)")

log_file.close()
print("\n💾 Log: logs/training_log.txt")

# =========================
# Gráficas
# =========================
fig = plt.figure(figsize=(20, 16))
x_axis = list(range(WINDOW, episode + 1, WINDOW))

# 1. Win Rate con tendencia
ax1 = plt.subplot(4, 2, 1)
ax1.plot(x_axis, dqn_win_rate, linewidth=2, color='green', marker='o', markersize=3)
ax1.axhline(y=25, color='r', linestyle='--', label='Random (25%)')
if best_win_rate > 0:
    ax1.axhline(y=best_win_rate, color='gold', linestyle='--', label=f'Best: {best_win_rate:.1f}%')
ax1.set_xlabel('Episodios')
ax1.set_ylabel('Win Rate (%)')
ax1.set_title('📈 Win Rate DQN (DETECCIÓN DE OLVIDO)')
ax1.grid(True, alpha=0.3)
ax1.legend()
ax1.set_ylim(0, 100)

# 2. Reward
ax2 = plt.subplot(4, 2, 2)
ax2.plot(x_axis, average_rewards, linewidth=2, color='blue')
ax2.set_xlabel('Episodios')
ax2.set_ylabel('Avg Reward')
ax2.set_title('💰 Reward Promedio')
ax2.grid(True, alpha=0.3)

# 3. Loss
ax3 = plt.subplot(4, 2, 3)
ax3.plot(x_axis, average_losses, linewidth=2, color='red')
ax3.set_xlabel('Episodios')
ax3.set_ylabel('Avg Loss')
ax3.set_title('📉 Loss')
ax3.grid(True, alpha=0.3)

# 4. Epsilon
ax4 = plt.subplot(4, 2, 4)
ax4.plot(range(1, episode + 1), epsilon_history, linewidth=1, color='purple', alpha=0.7)
ax4.set_xlabel('Episodios')
ax4.set_ylabel('Epsilon')
ax4.set_title('🎲 Exploration (Decay Lento)')
ax4.grid(True, alpha=0.3)

# 5. Q-values
ax5 = plt.subplot(4, 2, 5)
ax5.plot(range(1, episode + 1), q_values_history, linewidth=1, color='cyan', alpha=0.7)
ax5.set_xlabel('Episodios')
ax5.set_ylabel('Avg Max Q')
ax5.set_title('🎯 Q-values')
ax5.grid(True, alpha=0.3)
ax5.axhline(y=10, color='orange', linestyle='--', label='Saludable')
ax5.legend()

# 6. Victorias totales
ax6 = plt.subplot(4, 2, 6)
colors = ['green', 'red', 'orange', 'brown']
bars = ax6.bar(win_counts_total.keys(), win_counts_total.values(), color=colors)
ax6.set_ylabel('Victorias')
ax6.set_title('🏆 Total')
ax6.grid(axis='y', alpha=0.3)
for bar in bars:
    height = bar.get_height()
    ax6.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(height)}', ha='center', va='bottom')

# 7. Evolución victorias
ax7 = plt.subplot(4, 2, 7)
for i, (name, victories) in enumerate(victories_window.items()):
    ax7.plot(x_axis, victories, linewidth=2, color=colors[i], label=name, marker='o', markersize=3)
ax7.set_xlabel('Episodios')
ax7.set_ylabel('Victorias Acumuladas')
ax7.set_title('📊 Evolución')
ax7.grid(True, alpha=0.3)
ax7.legend()

# 8. Detección de olvido (pendiente del win rate)
ax8 = plt.subplot(4, 2, 8)
if len(dqn_win_rate) > 1:
    win_rate_deltas = [dqn_win_rate[i] - dqn_win_rate[i-1] for i in range(1, len(dqn_win_rate))]
    ax8.plot(x_axis[1:], win_rate_deltas, linewidth=2, color='purple', marker='o', markersize=3)
    ax8.axhline(y=0, color='gray', linestyle='-', linewidth=1)
    ax8.axhline(y=-5, color='red', linestyle='--', label='Umbral olvido')
    ax8.set_xlabel('Episodios')
    ax8.set_ylabel('Δ Win Rate (%)')
    ax8.set_title('🚨 Detector de Olvido Catastrófico')
    ax8.grid(True, alpha=0.3)
    ax8.legend()

plt.tight_layout()
plt.savefig('logs/training_results.png', dpi=150, bbox_inches='tight')
print("💾 Gráficas: logs/training_results.png")
plt.show()

env.close()

dqn1.save("models/final_model.pt")
print("💾 Modelo final guardado")