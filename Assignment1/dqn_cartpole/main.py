"""Hyperparameter search for the DQN agents.

Every (agent, hyperparameter combination, run) is trained in its own process. Everything it
produces goes into its own folder, so it is easy to find later:

    results/<agent>/<combination>/run<i>_log.txt                 hyperparameters + everything printed
    results/<agent>/<combination>/run<i>_loss_per_step.png
    results/<agent>/<combination>/run<i>_reward_per_episode.png
    results/<agent>/<combination>/run<i>_episode_rewards.npy     raw per-episode training rewards
    results/<agent>/<combination>/run<i>_result.json             final numbers of the run
    results/summary.csv                                          one row per finished run, all agents

Runs that already have a result.json are skipped (SKIP_FINISHED), so an interrupted search can
simply be started again.
"""
import csv
import glob
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product

import gymnasium as gym
import numpy as np

from dqn_agent import DQNAgent
from imp_dqn import ImprovedDQNAgent
from hyperparameter import *
import plots

AGENTS = {
    # name: (class, folder name)
    "DQN": (DQNAgent, "dqn"),
    "Improved DQN": (ImprovedDQNAgent, "improved_dqn"),
}
RUN_AGENTS = ["DQN", "Improved DQN"]  # remove one to search only the other

# Values to try, separately for each agent. Every combination of the four lists is run.
# A key that is left out uses its value from hyperparameter.py.
SEARCH = {
    "DQN": {
        "gamma":         [0.99, 0.98, 0.995],
        "min_epsilon":   [0.005, 0.01],
        "lr":            [1e-3, 5e-4],
        "epsilon_decay": [0.998, 0.995],
    },
    "Improved DQN": {
        "gamma":         [0.99, 0.98, 0.995],
        "min_epsilon":   [0.005, 0.01],
        "lr":            [1e-3, 5e-4],
        "epsilon_decay": [0.998, 0.995],
    },
}

N_RUNS = 5                  # independent runs per combination
N_TEST_EPISODES = 10        # greedy episodes used to evaluate each run's final policy
SKIP_FINISHED = True        # skip runs whose result.json already exists
MAX_WORKERS = None          # None = number of CPU cores minus 2
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

# hyperparameter.py name for every searched key
DEFAULTS = {"gamma": GAMMA, "min_epsilon": MIN_EPSILON, "lr": LR, "epsilon_decay": DECAY_RATE}


def combinations(name):
    grid = {key: SEARCH[name].get(key, [default]) for key, default in DEFAULTS.items()}
    return [dict(zip(grid, values)) for values in product(*grid.values())]


def combo_name(hp):
    return f"gamma{hp['gamma']:g}_minEps{hp['min_epsilon']:g}_lr{hp['lr']:g}_decay{hp['epsilon_decay']:g}"


def run_dir(name, hp):
    return os.path.join(RESULTS_DIR, AGENTS[name][1], combo_name(hp))


def result_path(name, hp, run):
    return os.path.join(run_dir(name, hp), f"run{run}_result.json")


class _Prefixed:
    """Prefixes every printed line, so output of parallel workers stays readable."""
    def __init__(self, stream, prefix):
        self.stream, self.prefix, self.at_line_start = stream, prefix, True

    def write(self, text):
        for chunk in text.splitlines(keepends=True):
            if self.at_line_start:
                self.stream.write(self.prefix)
            self.stream.write(chunk)
            self.at_line_start = chunk.endswith("\n")
        return len(text)

    def flush(self):
        self.stream.flush()


class _Tee:
    """Writes everything to all the given streams (console and log file)."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
        return len(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def train_and_report(name, hp, run):
    """Trains one agent, evaluates it, saves figures + log + result. Runs in its own process."""
    out_dir = run_dir(name, hp)
    os.makedirs(out_dir, exist_ok=True)
    log = open(os.path.join(out_dir, f"run{run}_log.txt"), "w", encoding="utf-8")
    real_stdout = sys.stdout
    console = _Prefixed(real_stdout, f"[{AGENTS[name][1]} {combo_name(hp)} #{run}] ")
    sys.stdout = _Tee(console, log)
    try:
        return _train_and_report(name, hp, run, out_dir)
    finally:
        sys.stdout = real_stdout
        log.close()


def _train_and_report(name, hp, run, out_dir):
    print(f"=== {name} (run {run}/{N_RUNS}) ===")
    print("Hyperparameters:")
    for key, value in {**hp, "epsilon_0": EPSILON, "memory_cap": MEMORY_CAP,
                       "target_update_c": C, "n_episodes": N_EPISODE}.items():
        print(f"  {key} = {value}")

    env = gym.make("CartPole-v1")
    agent = AGENTS[name][0](
        state_size=env.observation_space.shape[0],
        action_size=env.action_space.n,
        gamma=hp["gamma"],
        epsilon_0=EPSILON,
        epsilon_min=hp["min_epsilon"],
        epsilon_decay=hp["epsilon_decay"],
        lr=hp["lr"],
        memory_cap=MEMORY_CAP,
        c=C
    )
    agent.train_agent(n_episodes=N_EPISODE, env=env)
    env.close()

    test_rewards = agent.evaluate(env=env, n_episodes=N_TEST_EPISODES)

    solved_at = agent.episodes_to_solve(threshold=475.0, window=100)
    if solved_at is not None:
        print(f"Solved: average reward >= 475.0 over 100 consecutive episodes after {solved_at} episodes")
    else:
        print(f"Not solved: average reward >= 475.0 over 100 consecutive episodes was not reached in {N_EPISODE} episodes")

    last500 = float(np.mean(agent.episode_rewards[-500:]))
    print(f"Average training reward over the last 500 episodes: {last500:.1f}")

    plots.plot_loss_per_step(agent.losses, filename=f"run{run}_loss_per_step.png", out_dir=out_dir, label=name)
    plots.plot_reward_per_episode(agent.episode_rewards, filename=f"run{run}_reward_per_episode.png",
                                  out_dir=out_dir, label=name)
    np.save(os.path.join(out_dir, f"run{run}_episode_rewards.npy"), np.asarray(agent.episode_rewards))

    result = {
        "agent": name, "run": run, **hp,
        "test_mean": float(np.mean(test_rewards)), "test_std": float(np.std(test_rewards)),
        "test_min": float(np.min(test_rewards)), "test_max": float(np.max(test_rewards)),
        "solved_at": solved_at, "last500_mean": last500,
        "folder": os.path.relpath(out_dir, RESULTS_DIR),
    }
    # written last: its existence marks the run as finished
    with open(result_path(name, hp, run), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return result


def write_summary():
    """results/summary.csv: one row per finished run of every agent, best test mean first."""
    rows = []
    for path in glob.glob(os.path.join(RESULTS_DIR, "*", "*", "run*_result.json")):
        with open(path, encoding="utf-8") as f:
            rows.append(json.load(f))
    rows.sort(key=lambda r: (r["agent"], -r["test_mean"], -r["last500_mean"]))
    fields = ["agent", "gamma", "min_epsilon", "lr", "epsilon_decay", "run", "test_mean", "test_std",
              "test_min", "test_max", "solved_at", "last500_mean", "folder"]
    with open(os.path.join(RESULTS_DIR, "summary.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    jobs = []
    for name in RUN_AGENTS:
        for hp in combinations(name):
            for run in range(1, N_RUNS + 1):
                if SKIP_FINISHED and os.path.exists(result_path(name, hp, run)):
                    continue
                jobs.append((name, hp, run))

    total = sum(len(combinations(name)) * N_RUNS for name in RUN_AGENTS)
    print(f"{total} runs in the search, {total - len(jobs)} already finished, {len(jobs)} to run")
    if not jobs:
        sys.exit(0)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    workers = min(len(jobs), MAX_WORKERS or max(1, (os.cpu_count() or 2) - 2))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(train_and_report, *job): job for job in jobs}
        for i, future in enumerate(as_completed(futures), 1):
            name, hp, run = futures[future]
            try:
                result = future.result()
            except Exception as error:
                print(f"[{i}/{len(jobs)}] FAILED {name} {combo_name(hp)} run {run}: {error!r}")
                continue
            solved = result["solved_at"] if result["solved_at"] is not None else "not solved"
            print(f"[{i}/{len(jobs)}] done {name} {combo_name(hp)} run {run}: "
                  f"test mean {result['test_mean']:.1f}, solved {solved}")
            write_summary()

    print(f"\nResults are in {RESULTS_DIR}")
