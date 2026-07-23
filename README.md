# D.E.F.E.C.T.
**Deep Expert Framework for Evaluating Combinatorial Tasks**

An RL agent framework for playing Slay the Spire, built on top of [MiniSTS](https://github.com/iambb5445/MiniSTS). Phase 1 benchmarks learned agents (DQN and PPO) against rule-based, search-based, and LLM baselines on a progressive multi-fight combat benchmark.

---

## project structure

```
DEFECT/MiniSTS/
├── main.py                        # dqn training loop
├── train_ppo.py                   # ppo training loop
├── evaluate.py                    # evaluation suite (all agents)
├── plot_results.py                # visualization dashboard
├── training_log.json              # generated during dqn training
├── training_log_ppo.json          # generated during ppo training
├── eval_results.json              # generated during evaluation
├── dqn_model.pt                   # saved dqn weights
├── ppo_model.pt                   # saved ppo weights
├── game.py                        # minists game state
├── battle.py                      # minists battle loop
├── agent.py                       # player and enemy definitions
├── card.py                        # card definitions
├── ggpa/
│   ├── dqn_agent.py               # dqn bot (state vector + action selection)
│   ├── ppo_agent.py               # ppo bot (inherits state vector from dqn bot)
│   ├── always_attack.py           # aggressive baseline
│   ├── random_bot.py              # random baseline
│   ├── backtrack.py               # search-based baseline (depth 3)
│   ├── chatgpt_bot.py             # llm baseline
│   └── rl_algos/
│       ├── __init__.py
│       ├── dqn.py                 # dqn network, replay buffer, optimizer
│       ├── ppo.py                 # ppo actor-critic, rollout buffer, gae, clipped update
│       └── training_logger.py     # logs training metrics to json
```

---

## setup

**requirements:** python 3.11.4+, ubuntu (or wsl), amd gpu with rocm or nvidia gpu with cuda

```bash
# clone and enter
git clone https://github.com/KELECHIDAVIS/DEFECT.git
cd DEFECT/MiniSTS

# create virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# install dependencies
pip install -r requirements.txt

# install pytorch with rocm (amd gpu)
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm7.2 --no-cache-dir

# install pytorch with cuda (nvidia gpu)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --no-cache-dir

# verify gpu is detected
python3.11 -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

**amd gpu permissions (required on linux):**
```bash
sudo usermod -a -G render,video $USER
# then log out and back in
```

**if gpu still not detected (rx 7900 xtx):**
```bash
echo 'export HSA_OVERRIDE_GFX_VERSION=11.0.0' >> .venv/bin/activate
source .venv/bin/activate
```

---

## training

**dqn:**
```bash
# train dqn agent (edit main.py to switch agents)
python3.11 main.py
```

**ppo:**
```bash
# train ppo agent (3000 episodes on gpu, 50 on cpu by default)
python3.11 train_ppo.py

# short run for testing
python3.11 train_ppo.py --episodes 100
```

both agents share the identical 266-dim state encoding, action mask, reward
(+1 + hp/80 on fight win, -1 on loss), and four-fight benchmark -- any
performance difference is attributable to the algorithm.

the key training difference: dqn is off-policy (replay buffer, per-step
updates, epsilon-greedy exploration), ppo is on-policy (collects 8 full
episodes, runs the clipped surrogate update over that batch, discards it,
and repeats -- exploration comes from sampling the stochastic policy).

training output per episode:
```
episode 1/600: LOSE | hp: 0/80 | reward: -0.823 | steps: 47 | 0.31s
episode 10/600: WIN  | hp: 34/80 | reward: 1.241 | steps: 38 | 0.28s
...
episode 10/600 summary: win rate (last 20): 20.0% | avg reward: 0.412 | total win rate: 10.0%
```

training logs saved to `training_log.json` (dqn) and `training_log_ppo.json`
(ppo) every 10 episodes. models saved to `dqn_model.pt` and `ppo_model.pt`.

**switching agents in main.py:**
```python
agent = DQNBot()           # dqn (trains)
# agent = AlwaysAttackBot()  # aggressive baseline
# agent = BacktrackBot(3, False)  # search baseline
# agent = RandomBot()        # random baseline
```

---

## evaluation

run all agents against 50 identical seeded scenarios:

```bash
python3.11 evaluate.py
```

run specific agent:
```bash
python3.11 evaluate.py --agent dqn
python3.11 evaluate.py --agent ppo
python3.11 evaluate.py --agent always_attack
python3.11 evaluate.py --agent backtrack
python3.11 evaluate.py --agent random
```

run more episodes for tighter estimates:
```bash
python3.11 evaluate.py --episodes 100
```

select a battle suite (default: four):
```bash
python3.11 evaluate.py --battles single   # jawworm only
python3.11 evaluate.py --battles four     # full progressive benchmark
```

each run starts fresh -- an existing `eval_results.json` is removed at the
start so every agent in the run is evaluated with the current models.

rl agents evaluate deterministically: dqn takes the greedy argmax over
q-values, ppo takes the argmax of the policy distribution.

**llm agents (requires openai api key):**
```bash
export OPENAI_API_KEY="your-key-here"
python3.11 evaluate.py --agent llm       # gpt-3.5-turbo, no chain of thought
python3.11 evaluate.py --agent llm_cot   # gpt-3.5-turbo, chain of thought
```

get a key at platform.openai.com. 50 episodes costs under $1 with gpt-3.5-turbo.

**evaluation summary table printed after each run (single-fight suite shown):**

| Agent | Win% | Avg HP | Avg Dmg | Avg Turns | HP (win) |
|---|---|---|---|---|---|
| Random | 76.0% | 26.3 | 53.7 | 24.6 | 34.6 |
| AlwaysAttack | 100.0% | 55.0 | 25.0 | 9.7 | 55.0 |
| Backtrack (depth 3) | 100.0% | 53.6 | 26.4 | 12.1 | 53.6 |
| **DQN** | **100.0%** | **58.4** | **21.6** | **10.6** | **58.4** |
| PPO | -- | -- | -- | -- | -- |
| LLM | 100.0% | 51.6 | 28.4 | 16.1 | 51.6 |

*(ppo row pending full 3000-episode training run)*

---

## visualization

generate comparison dashboard (requires eval_results.json):

```bash
python3.11 plot_results.py
```

save plots to png:
```bash
python3.11 plot_results.py --save
# saves to plots/defect_evaluation_dashboard.png
# saves to plots/defect_training_curve.png
```

use a different results file:
```bash
python3.11 plot_results.py --input my_results.json
```

**dashboard panels:**
- win rate bar chart (with skilled/pro player reference lines)
- hp distribution histogram for all agents (mirrors llm paper figures)
- average hp: all episodes vs winning episodes
- average damage taken
- turn count violin plot
- running win rate over evaluation episodes
- dqn training curve (reward + win rate over training episodes)

---

## ssh remote development

connect from windows laptop to desktop for training:

```bash
# on desktop -- enable ssh
sudo apt install openssh-server
sudo systemctl enable ssh && sudo systemctl start ssh

# install tailscale for access outside home network
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up
```

in vs code (windows): install `Remote - SSH` extension, then `Ctrl+Shift+P` > `Remote-SSH: Connect to Host` > enter `kelechi@<tailscale-ip>`

**keep desktop awake during training:**
```bash
sudo systemctl mask sleep.target suspend.target hibernate.target hybrid-sleep.target
```

**run training in background (survives disconnects):**
```bash
tmux new -s training
python3.11 main.py
# detach: Ctrl+B then D
# reattach later: tmux attach -t training
```

---

## benchmarks

reference values from the llm paper (Bateni & Whitehead, FDG 2024):

| agent | avg final hp | notes |
|---|---|---|
| random | 5.3 | starter deck, single enemy |
| backtrack depth 3 | 25.94 | starter deck, single enemy |
| llm (none) | 11.72 | gpt-3.5, no cot |
| llm cot | 23.36 | gpt-3.5, chain of thought |

your evaluation uses a more realistic setup: 5 cards drawn per turn (vs 3 in paper), same starter deck, same ironclad hp (80), same jawworm enemy.

player win rate benchmarks from design doc:
- skilled roguelite player: 20-50%
- pro player: 50%+

---

## adding new agents

1. create `ggpa/your_agent.py` extending `GGPA`
2. implement `choose_card`, `choose_agent_target`, `choose_card_target`
3. add to `build_agents()` in `evaluate.py`
4. add a color entry in `AGENT_COLORS` in `plot_results.py`

for a new rl algorithm, follow the ppo pattern: put the algorithm in
`ggpa/rl_algos/`, subclass `DQNBot` in the agent file to inherit the shared
state encoding, and add a dedicated training script.

---

## reference docs

- `state_vectors_reference.txt` -- full 266-dim state tensor specification
- `D_E_F_E_C_T_Design_Reference_Doc.docx` -- full project design, research question, publication plan
- [MiniSTS repo](https://github.com/iambb5445/MiniSTS) -- environment documentation
- [pytorch dqn tutorial](https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html) -- dqn implementation reference
- [cleanrl](https://github.com/vwxyzjn/cleanrl) -- clean ppo/dqn reference implementations
