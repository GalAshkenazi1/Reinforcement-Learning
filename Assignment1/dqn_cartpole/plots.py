import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # figures are only saved, never shown (also safe inside worker processes)
import matplotlib.pyplot as plt

FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")


def _save(fig, filename, out_dir=None):
    out_dir = out_dir or FIGURES_DIR
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def _moving_average(values, window):
    values = np.asarray(values, dtype=float)
    if window <= 1 or len(values) < window:
        return None
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def plot_loss_per_step(losses, window=100, filename="dqn_loss_per_step.png", out_dir=None, label="DQN"):
    """losses: loss value of every training (gradient) step.
    Raw loss is drawn faintly, with a moving average over `window` steps on top."""
    fig, ax = plt.subplots(figsize=(10, 4))
    steps = np.arange(1, len(losses) + 1)
    ax.plot(steps, losses, linewidth=0.5, alpha=0.4, label="Loss")

    avg = _moving_average(losses, window)
    if avg is not None:
        ax.plot(steps[window - 1:], avg, linewidth=1.5, label=f"Moving average ({window})")
        ax.legend()

    ax.set_title(f"{label} loss per training step")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Loss (MSE)")
    ax.grid(alpha=0.3)
    _save(fig, filename, out_dir)


def plot_reward_per_episode(rewards, window=50, filename="dqn_reward_per_episode.png", out_dir=None, label="DQN"):
    """rewards: total reward of every training episode.
    Raw reward is drawn faintly, with a moving average over `window` episodes on top."""
    fig, ax = plt.subplots(figsize=(10, 4))
    episodes = np.arange(1, len(rewards) + 1)
    ax.plot(episodes, rewards, linewidth=0.8, alpha=0.5, label="Total reward")

    avg = _moving_average(rewards, window)
    if avg is not None:
        ax.plot(episodes[window - 1:], avg, linewidth=1.8, label=f"Moving average ({window})")
        ax.legend()

    ax.set_title(f"{label} total reward per training episode")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total reward")
    ax.grid(alpha=0.3)
    _save(fig, filename, out_dir)
