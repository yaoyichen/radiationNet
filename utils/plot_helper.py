import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


def plot_RTM(predicts, targets, filename, sample_index):

    predicts = predicts.cpu().detach().numpy()
    targets = targets.cpu().detach().numpy()

    fig = plt.figure(tight_layout=True, figsize=(12, 5),)
    gs = gridspec.GridSpec(3, 4)
    # ref: https://matplotlib.org/stable/gallery/subplots_axes_and_figures/align_labels_demo.html#sphx-glr-gallery-subplots-axes-and-figures-align-labels-demo-py
    name_dict = {0: {"name": "swuflx", "plotname": "short_up"},
                 1: {"name": "swdflx", "plotname": "short_down"},
                 2: {"name": "lwuflx", "plotname": "long_up"},
                 3: {"name": "lwdflx", "plotname": "long_down"}}
    for variable in range(4):
        ax = fig.add_subplot(gs[0, variable])
        ax.plot(predicts[sample_index, variable, :], label="predict")
        ax.plot(targets[sample_index, variable, :], label="true")
        ax.set_title(name_dict[variable]["plotname"], fontsize=10)
        ax.set_xticks(np.arange(0, 56, 25))
        # if(variable < 2):
        #     ax.set_yticks(np.arange(0, 501, 100))
        #     ax.set_ylim([-1, 500])
        # else:
        #     ax.set_yticks(np.arange(0, 501, 100))
        #     ax.set_ylim([-1, 500])
        ax.set_ylabel("Flux rmse W m^{-2}")
        ax.legend()

    plt.savefig(filename, dpi=500, bbox_inches='tight')
    plt.close(fig)
    plt.clf()
    plt.cla()


def plot_HeatRate(swhr_predict, swhr_target,
                  lwhr_predict, lwhr_target, filename, sample_index):

    swhr_predict = swhr_predict.cpu().detach().numpy()
    swhr_target = swhr_target.cpu().detach().numpy()
    lwhr_predict = lwhr_predict.cpu().detach().numpy()
    lwhr_target = lwhr_target.cpu().detach().numpy()

    fig = plt.figure(tight_layout=True, figsize=(7, 5),)
    gs = gridspec.GridSpec(3, 2)

    ax = fig.add_subplot(gs[0, 0])
    ax.plot(swhr_predict[sample_index, 0, :], label="predict")
    ax.plot(swhr_target[sample_index, 0, :], label="true")
    ax.set_title("sw heat rate", fontsize=10)
    ax.set_ylabel("Heat rate K d^{--1}")
    ax.legend()

    ax = fig.add_subplot(gs[0, 1])
    ax.plot(lwhr_predict[sample_index, 0, :], label="predict")
    ax.plot(lwhr_target[sample_index, 0, :], label="true")
    ax.set_title("lw heat rate", fontsize=10)
    ax.set_ylabel("Heat rate K d^{--1}")
    ax.legend()

    plt.savefig(filename, dpi=500, bbox_inches='tight')
    plt.close(fig)
    plt.clf()
    plt.cla()
