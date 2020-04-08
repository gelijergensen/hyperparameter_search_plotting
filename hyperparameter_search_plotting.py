import math
import numpy as np
import os.path

import matplotlib as mpl

mpl.use("agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def _get_gridspec_size(num_subfigures):
    num_cols = int(math.ceil(math.sqrt(num_subfigures)))
    num_rows = int(math.ceil(num_subfigures / num_cols))
    return num_rows, num_cols


def gridspec_and_subplot_axes_generator(fig, num_parameters):
    num_subfigures = (num_parameters * (num_parameters - 1)) // 2
    n_rows, n_cols = _get_gridspec_size(num_subfigures)
    gs = GridSpec(n_rows, n_cols, figure=fig)

    def subplot_axes_generator():
        for i in range(num_subfigures):
            row = i // n_cols
            col = i % n_cols
            yield fig.add_subplot(gs[row, col])

    return gs, subplot_axes_generator


def get_plottable_score(scores_list):
    """This is the key function. It takes all observed scores and returns a 
    value which we plot. Generally, this is the optimal score from the list
    (i.e. the max), but it could be say, the standard deviation of the score"""
    return np.max(scores_list)


def get_best_plottable_score_index(plottable_scores):
    """This is the second key function. Here, we take a collection of all the 
    plotted scores and return the index of the best one"""
    return np.argmax(plottable_scores)


def get_data_by_parameter_values(param1, param2, parameter_dicts, scores):
    """Retrieves the optimal score for each combination of parameter values

    :param param1: string for the first parameter name
    :param param2: string for the second parameter name
    :param searched_points: the list of all points which were searched
    :returns: numpy arrays for param1 values and param2 values, and a list of 
        lists of scores for those values
    """
    all_p1_vals = list()
    all_p2_vals = list()
    all_scores = list()
    for params, score in zip(parameter_dicts, scores):
        param1_val = params[param1]
        param2_val = params[param2]
        for i, (p1_val, p2_val) in enumerate(zip(all_p1_vals, all_p2_vals)):
            if p1_val == param1_val and p2_val == param2_val:
                all_scores[i].append(score)
                break
        else:
            all_p1_vals.append(param1_val)
            all_p2_vals.append(param2_val)
            all_scores.append([score])
    return np.array(all_p1_vals), np.array(all_p2_vals), scores


def subplot_parameter_names_generator(parameter_names):
    for i, p1_name in enumerate(parameter_names):
        for p2_name in parameter_names[i + 1 :]:
            yield p1_name, p2_name


def get_data_to_plot(p1_name, p2_name, parameter_dicts, scores):
    p1_vals, p2_vals, scores_lists = get_data_by_parameter_values(
        p1_name, p2_name, parameter_dicts, scores
    )
    scores_to_plot = np.array(list(map(get_plottable_score, scores_lists)))
    return p1_vals, p2_vals, scores_to_plot


def plot_datapoints(
    ax, xvals, yvals, scores, background_color="grey", cmap=plt.get_cmap("hot")
):
    ax.set_facecolor(background_color)
    try:
        levels = np.linspace(np.min(scores), np.max(scores))
        ax.tricontourf(xvals, yvals, scores, cmap=cmap, levels=levels)
    except:
        # If something goes wrong, we revert to a scatter plot
        ax.scatter(xvals, yvals, c=scores, cmap=cmap)
    return ax


def plot_best_value(
    ax, best_x, best_y, color="blue", marker="o", linestyle="--"
):
    ax.axvline(best_x, color=color, linestyle=linestyle)
    ax.axhline(best_y, color=color, linestyle=linestyle)
    ax.plot(best_x, best_y, color=color, marker=marker)
    return ax


def set_subplot_limits(ax, xlimits, ylimits):
    ax.set_xlim(*xlimits)
    ax.set_ylim(*ylimits)
    return ax


def set_subplot_labels(ax, xlabel=None, ylabel=None, title=None):
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    return ax


def save_figure(fig, filename, directory="."):
    filepath = os.path.join(directory, filename)
    print(f"Saving file to {filepath}")
    fig.savefig(filepath, bbox_inches="tight", pad_inches=0.2)


def finalize_figure(
    fig, gs, supertitle=None, tight_layout=True, colorbar_label=None
):
    if supertitle is not None:
        suptitle = fig.suptitle(supertitle)

    # Note, this line can fail, which is why we have a guard for it
    if tight_layout:
        gs.tight_layout(fig, rect=[0, 0.08, 1, 0.95], h_pad=0.5)

    # handle the colorbar (for the first image in the first axis)
    first_plot = fig.axes[0].collections[0]
    cax = fig.add_axes([0.09, 0.06, 0.84, 0.02])
    fig.colorbar(first_plot, cax=cax, orientation="horizontal")
    if colorbar_label is not None:
        cax.set_xlabel(colorbar_label)

    return fig


def plot_everything(parameter_dicts, scores, parameter_names=None):
    if parameter_names is None:
        parameter_names = list(parameter_dicts[0].keys())

    fig = plt.figure()

    gs, axes_gen = gridspec_and_subplot_axes_generator(
        fig, len(parameter_names)
    )

    for axis, (param1_name, param2_name) in zip(
        axes_gen(), subplot_parameter_names_generator(parameter_names)
    ):
        xvals, yvals, plottable_scores = get_data_to_plot(
            param1_name, param2_name, parameter_dicts, scores
        )
        best_index = get_best_plottable_score_index(plottable_scores)

        axis = plot_datapoints(axis, xvals, yvals, plottable_scores)
        axis = plot_best_value(axis, xvals[best_index], yvals[best_index])
        axis = set_subplot_labels(
            axis,
            xlabel=param1_name,
            ylabel=param2_name,
            title=f"{param1_name} and {param2_name}",
        )

    finalize_figure(
        fig, gs, supertitle="Look ma, it worked!", colorbar_label="score"
    )

    return fig


# TODO:
# configuration
# - objects? long set of kwargs?
# handle normalization of color in tricolorf!
# convert to an object


if __name__ == "__main__":

    def score(**params):
        return sum(params.values())

    num_points = 3 * 3 * 2 * 2

    all_params = [
        {"a": i, "b": 3 - 2 * i, "c": i * i, "d": 3 - 2 * i * i}
        for i in range(num_points)
    ]
    all_scores = [score(**params) for params in all_params]

    fig = plot_everything(all_params, all_scores)
    save_figure(fig, "test.png")
