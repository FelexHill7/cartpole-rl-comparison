import os
import numpy as np
import matplotlib.pyplot as plt

def allow_empty_plot(plt, arr, *args, **kwargs):
    if isinstance(arr, list):
        arr = list(enumerate(arr))
        arr = list(filter(lambda x: x[1] is not None, arr))
    plt.plot([x[0] for x in arr], [x[1] for x in arr], *args, **kwargs)


def plot_rewards(rewards, title, path):
    plt.clf()
    allow_empty_plot(plt, rewards)
    plt.xlabel("episode")
    plt.ylabel("reward")
    plt.title(title)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path)


def plot_loss(losses, title, path):
    if all(loss is None for loss in losses):
        return
    all_dict = all(isinstance(loss, (dict, type(None))) for loss in losses)
    all_number = all(isinstance(loss, (float, int, type(None))) for loss in losses)
    assert all_dict or all_number
    plt.clf()
    if all_dict:
        for loss in losses:
            if loss:
                keys = list(loss.keys())
                break
        fig, axs = plt.subplots(ncols=len(keys), figsize=(5 * len(keys), 4))
        losses = list(map(lambda x: x if x is not None else {key: None for key in keys}, losses))
        for i, key in enumerate(keys):
            allow_empty_plot(axs[i], [loss[key] for loss in losses], label=key)
            axs[i].set_xlabel("episode")
            axs[i].set_ylabel("loss")
            axs[i].set_title(f"{title} - {key}")
    else:
        allow_empty_plot(plt, losses, label='loss')
        plt.xlabel("episode")
        plt.ylabel("loss")
        plt.title(title)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path)


def plot_compare(dict_of_rewards, title, path):
    plt.clf()
    for name, rewards in dict_of_rewards.items():
        allow_empty_plot(plt, rewards, label=name)
    plt.xlabel("episode")
    plt.ylabel("reward")
    plt.title(title)
    plt.legend()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path)
