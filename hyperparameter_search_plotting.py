import math
import numpy as np
import os.path

import matplotlib as mpl

mpl.use("agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


############### Handy plotting utilities ###################


def create_gridspec(fig, num_subfigures):
    """Intelligently creates a GridSpec which has at least as many columns as
    rows of subplots and minimizes whitespace

    :param fig: figure object
    :param num_subfigures: integer for the number of subfigures
    :returns: a GridSpec object
    """
    num_cols = int(math.ceil(math.sqrt(num_subfigures)))
    num_rows = int(math.ceil(num_subfigures / num_cols))
    return GridSpec(num_rows, num_cols, figure=fig)


def set_axis_limits(ax, xlimits=None, ylimits=None):
    """Sets the x- and y-boundaries of the axis (if provided)

    :param ax: axis object
    :param xlimits: a 2-tuple (lower_bound, upper_bound)
    :param ylimits: a 2-tuple (lower_bound, upper_bound)
    :returns: ax
    """
    if xlimits is not None:
        ax.set_xlim(*xlimits)
    if ylimits is not None:
        ax.set_ylim(*ylimits)
    return ax


def set_axis_labels(ax, xlabel=None, ylabel=None, title=None):
    """Sets the labels for a single axis

    :param ax: axis object
    :param xlabel: string for the x label of the axis
    :param ylabel: string for the y label of the axis
    :param title: title for the axis
    :returns: ax
    """
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    return ax


def save_figure(fig, filename, directory="."):
    """Saves a figure to file

    :param fig: figure object
    :param filename: string for the name of the file
    :param directory: string for the directory to save in. Defaults to the 
        current directory
    """
    filepath = os.path.join(directory, filename)
    print(f"Saving file to {filepath}")
    fig.savefig(filepath, bbox_inches="tight", pad_inches=0.2)


def finalize_figure(
    fig,
    gs,
    supertitle=None,
    tight_layout=True,
    with_colorbar=True,
    colorbar_label=None,
):
    """Takes a figure and gridspec and finalizes by possibly adding a 
    supertitle, compressing the subfigures, and adding a global colorbar

    :param fig: figure object
    :param gs: GridSpec for this figure
    :param supertitle: optional title for the figure
    :param tight_layout: whether to compress the whitespace between the 
        subfigures. Defaults to True
    :param with_colorbar: whether to add a global colorbar
    :param colorbar_label: optional label for the colorbar
    :returns: the figure
    """

    if supertitle is not None:
        suptitle = fig.suptitle(supertitle)
        top_edge = 0.95
    else:
        top_edge = 1

    # Note, this line can fail, which is why we have a guard for it
    if tight_layout:
        if with_colorbar:
            bottom_edge = 0.08
        else:
            bottom_edge = 0
        gs.tight_layout(fig, rect=[0, bottom_edge, 1, top_edge], h_pad=0.5)

    if with_colorbar:
        # handle the colorbar (for the first image in the first axis)
        first_plot = fig.axes[0].collections[0]
        cax = fig.add_axes([0.09, 0.06, 0.84, 0.02])
        fig.colorbar(first_plot, cax=cax, orientation="horizontal")
        if colorbar_label is not None:
            cax.set_xlabel(colorbar_label)

    return fig


############### Main plotting function #####################


def plot_hyperparameter_results(
    parameter_dicts,
    scores,
    parameter_names=None,
    scores_transform=np.max,
    best_index_retriever=np.argmax,
):
    if parameter_names is None:
        parameter_names = list(parameter_dicts[0].keys())

    parameter_name_pairs = list(
        _subplot_parameter_names_generator(parameter_names)
    )

    fig = plt.figure()

    gs = create_gridspec(fig, len(parameter_name_pairs))

    for axis, (param1_name, param2_name) in zip(
        _subplot_axes_generator(fig, gs), parameter_name_pairs
    ):
        xvals, yvals, plottable_scores = _get_data_to_plot(
            param1_name, param2_name, parameter_dicts, scores, scores_transform
        )
        # TODO
        best_index = best_index_retriever(plottable_scores)

        axis = _plot_scores(axis, xvals, yvals, plottable_scores)
        axis = _plot_best_value(axis, xvals[best_index], yvals[best_index])
        axis = set_axis_labels(
            axis,
            xlabel=param1_name,
            ylabel=param2_name,
            title=f"{param1_name} and {param2_name}",
        )

    finalize_figure(
        fig, gs, supertitle="Look ma, it worked!", colorbar_label="score"
    )

    return fig


############### Helper functions ###########################


def _get_data_by_parameter_values(param1, param2, parameter_dicts, scores):
    """Retrieves the scores for each combination of parameter values

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


def _get_data_to_plot(
    p1_name, p2_name, parameter_dicts, scores, scores_transform
):
    p1_vals, p2_vals, scores_lists = _get_data_by_parameter_values(
        p1_name, p2_name, parameter_dicts, scores
    )
    scores_to_plot = np.array(list(map(scores_transform, scores_lists)))
    return p1_vals, p2_vals, scores_to_plot


def _plot_best_value(
    ax, best_x, best_y, color="blue", marker="o", linestyle="--"
):
    ax.axvline(best_x, color=color, linestyle=linestyle)
    ax.axhline(best_y, color=color, linestyle=linestyle)
    ax.plot(best_x, best_y, color=color, marker=marker)
    return ax


def _plot_scores(
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


def _subplot_axes_generator(fig, gs):
    num_rows, num_cols = gs.get_geometry()
    num_subfigures_limit = num_rows * num_cols
    for i in range(num_subfigures_limit):
        row = i // num_cols
        col = i % num_cols
        yield fig.add_subplot(gs[row, col])


def _subplot_parameter_names_generator(parameter_names):
    for i, p1_name in enumerate(parameter_names):
        for p2_name in parameter_names[i + 1 :]:
            yield p1_name, p2_name


# TODO:
# configuration
# - objects? long set of kwargs?
# handle normalization of color in tricolorf!
# convert to an object

############### Test #######################################
if __name__ == "__main__":

    def score(**params):
        return sum(params.values())

    num_points = 3 * 3 * 2 * 2

    all_params = [
        {"a": i, "b": 3 - 2 * i, "c": i * i, "d": 3 - 2 * i * i}
        for i in range(num_points)
    ]
    all_scores = [score(**params) for params in all_params]

    fig = plot_hyperparameter_results(all_params, all_scores)
    save_figure(fig, "test.png")
