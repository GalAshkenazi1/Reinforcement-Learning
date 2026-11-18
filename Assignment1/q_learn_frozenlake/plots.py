import os
import numpy as np
import matplotlib.pyplot as plt

FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
ACTION_LABELS = ["Left", "Down", "Right", "Up"]


def _save(fig, filename):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    path = os.path.join(FIGURES_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def plot_q_tables(q_snapshots, filename="q_tables.png"):
    """q_snapshots: dict {episode: q_table}. One subplot per snapshot, values annotated."""
    n = len(q_snapshots)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 8), squeeze=False)
    axes = axes[0]
    vmin = min(q.min() for q in q_snapshots.values())
    vmax = max(q.max() for q in q_snapshots.values())

    for ax, (episode, q) in zip(axes, q_snapshots.items()):
        im = ax.imshow(q, cmap="viridis", aspect="auto", vmin=vmin, vmax=vmax)
        ax.set_title(f"Q-table after {episode} episodes")
        ax.set_xlabel("Action")
        ax.set_ylabel("State")
        ax.set_xticks(range(q.shape[1]), ACTION_LABELS[:q.shape[1]])
        ax.set_yticks(range(q.shape[0]))
        for i in range(q.shape[0]):
            for j in range(q.shape[1]):
                ax.text(j, i, f"{q[i, j]:.2f}", ha="center", va="center",
                        color="white", fontsize=7)

    fig.colorbar(im, ax=axes, shrink=0.8, label="Q-value")
    _save(fig, filename)


def plot_reward_per_episode(rewards, filename="reward_per_episode.png"):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(np.arange(1, len(rewards) + 1), rewards, linewidth=0.8)
    ax.set_title("Reward per episode")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total reward")
    ax.grid(alpha=0.3)
    _save(fig, filename)


def plot_avg_steps_to_goal(steps, window=100, filename="avg_steps_to_goal.png"):
    """steps: per-episode step count (already set to max steps if goal not reached).
    Plots the mean over the last `window` episodes, every `window` episodes."""
    steps = np.asarray(steps)
    x = np.arange(window, len(steps) + 1, window)
    y = [steps[e - window:e].mean() for e in x]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x, y, marker="o")
    ax.set_title(f"Average steps to goal (last {window} episodes)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Average steps")
    ax.grid(alpha=0.3)
    _save(fig, filename)
