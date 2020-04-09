import math
import numpy as np
import os.path

import matplotlib as mpl

mpl.use("agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FormatStrFormatter, NullLocator, MaxNLocator


############### Handy plotting utilities ###################


def save_figure(fig, filename, directory=".", dpi=100):
    """Saves a figure to file

    :param fig: figure object
    :param filename: string for the name of the file
    :param directory: string for the directory to save in. Defaults to the 
        current directory
    :param dpi: dpi for the saved figure
    """
    filepath = os.path.join(directory, filename)
    print(f"Saving file to {filepath}")
    fig.savefig(filepath, bbox_inches="tight", pad_inches=0.2, dpi=dpi)


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


def set_axis_tick_format(
    ax, xtickformat=None, ytickformat=None, xrotation=0, yrotation=0
):
    """Sets the formats for the ticks of a single axis

    :param ax: axis object
    :param xtickformat: optional string for the format of the x ticks
    :param ytickformat: optional string for the format of the y ticks
    :param xrotation: rotation angle of the x ticks. Defaults to 0
    :param yrotation: rotation angle of the y ticks. Defaults to 0
    :returns: ax
    """
    if xtickformat is not None:
        ax.xaxis.set_major_formatter(FormatStrFormatter(xtickformat))
    if ytickformat is not None:
        ax.yaxis.set_major_formatter(FormatStrFormatter(ytickformat))

    plt.setp(ax.get_xticklabels(), ha="right", rotation=xrotation)
    plt.setp(ax.get_yticklabels(), ha="right", rotation=yrotation)
    return ax


def set_supertitle(fig, supertitle=None):
    """Takes a figure and (possibly) adds a supertitle

    :param fig: figure object
    :param supertitle: optional string for the supertitle
    :returns: fig
    """
    if supertitle is not None:
        fig.suptitle(supertitle)
    return fig


############### Main plotting function #####################


def plot_hyperparameter_results(
    parameter_dicts,
    scores,
    parameter_names=None,
    scores_transform=np.max,
    title=None,
    colorbar_label=None,
    cmap=plt.get_cmap("hot"),
    untested_color="grey",
    axes_tick_format="%1.2g",
    colorbar_tick_format="%g",
    xtick_rotation=45,
    ytick_rotation=0,
    bounds=dict(),
    colorbar_bounds=None,
):
    """Plots the results from a hyperparameter search, given the parameter 
    names and values and the resulting scores

    :param parameter_dicts: a list of dictionaries with keys of the parameter 
        names and values of the parameter values
    :param scores: a list of the same length as parameter_dicts with the 
        resulting scores
    :param parameter_names: a list of parameter names to actually plot (useful
        for removing parameters which were not numeric). Defaults to the keys of
        the first dictionary in parameter_dicts
    :param scores_transform: function which takes as input a list of all the 
        scores seen by a particular combination of two parameter values (e.g. 
        for "param1"=3 and "param2"=4 and outputs a single value. Generally
        either the maximum or minimum value, but could be something like the
        standard deviation or mean. Defaults to numpy.max
    :param title: string for the title of the plot. Defaults to no title
    :param colorbar_label: string for the label of the colorbar. Defaults to no
        label
    :param cmap: Colormap to be used for assigning colors. Defaults to "hot"
    :param untested_color: Color to be used for points which weren't tested
    :param axes_tick_format: string for the format of the numbers in the
        axes. Defaults to "general format" with 2 significant figures
    :param colorbar_tick_format: string for the format of the numbers in the
        axes. Defaults to "general format"
    :param xtick_rotation: rotation angle for the x ticks. Defaults to diagonal
        from the bottom left to upper right
    :param ytick_rotation: rotation angle for the y ticks. Defaults to no 
        rotation
    :param bounds: a dictionary with keys of parameter names and values of 
        2-tuples (lower_bound, upper_bound). Intended for cropping individual
        parameter ranges plotted. All keys not provided are not bounded
    :param colorbar_bounds: a 2-tuple (lower_bound, upper_bound) for the minimal
        and maximal scores to bound the colorbar. Defaults to no bounding. NOTE:
        if no bounds are provided, then color scaling can differ between 
        subplots
    :returns: the resulting figure
    """
    if parameter_names is None:
        parameter_names = list(parameter_dicts[0].keys())

    parameter_name_pairs = list(
        _subplot_parameter_names_generator(parameter_names)
    )

    fig, axes = plt.subplots(
        nrows=len(parameter_names),
        ncols=len(parameter_names),
        sharex="col",
        sharey="row",
        gridspec_kw=dict(wspace=0, hspace=0, top=0.93),
        figsize=(2 * len(parameter_names), 2 * len(parameter_names)),
    )

    for axis, (param1_name, param2_name) in zip(
        axes.ravel(), parameter_name_pairs,
    ):
        xvals, yvals, plottable_scores = _get_data_to_plot(
            param1_name, param2_name, parameter_dicts, scores, scores_transform
        )
        axis = _plot_scores(
            axis,
            xvals,
            yvals,
            plottable_scores,
            cmap=cmap,
            background_color=untested_color,
            bounds=colorbar_bounds,
        )
        axis = set_axis_labels(axis, xlabel=param1_name, ylabel=param2_name)
        axis.label_outer()
        axis = set_axis_tick_format(
            axis,
            axes_tick_format,
            axes_tick_format,
            xrotation=xtick_rotation,
            yrotation=ytick_rotation,
        )
        axis = set_axis_limits(
            axis,
            xlimits=bounds.get(param1_name),
            ylimits=bounds.get(param2_name),
        )

    fig = set_supertitle(fig, title)

    cax = fig.colorbar(
        fig.axes[0].collections[0],  # this is the first subplot image
        ax=axes,
        orientation="vertical",
        fraction=0.05,
        format=FormatStrFormatter(colorbar_tick_format),
    )
    if colorbar_label is not None:
        cax.set_label(colorbar_label)
    fig.align_xlabels(axes)
    fig.align_ylabels(axes)

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


def _plot_scores(
    ax,
    xvals,
    yvals,
    scores,
    background_color="grey",
    cmap=plt.get_cmap("hot"),
    bounds=None,
):
    ax.set_facecolor(background_color)
    try:
        levels = np.linspace(np.min(scores), np.max(scores))
        if bounds is not None:
            ax.tricontourf(
                xvals,
                yvals,
                scores,
                cmap=cmap,
                levels=levels,
                vmin=bounds[0],
                vmax=bounds[1],
            )
        else:
            ax.tricontourf(xvals, yvals, scores, cmap=cmap, levels=levels)
    except:
        # If something goes wrong, we revert to a scatter plot
        if bounds is not None:
            ax.scatter(
                xvals,
                yvals,
                c=scores,
                cmap=cmap,
                vmin=bounds[0],
                vmax=bounds[1],
            )
        else:
            ax.scatter(xvals, yvals, c=scores, cmap=cmap)
    return ax


def _subplot_parameter_names_generator(parameter_names):
    for p1_name in parameter_names:
        for p2_name in parameter_names:
            yield p2_name, p1_name


############### Test #######################################
if __name__ == "__main__":

    def score(**params):
        return sum(params.values())

    num_points = 3 * 3 * 2 * 2

    all_params = [
        {"a": i, "b": 3 - 2 * i - 120, "c": i * i, "d": 3 - 2 * i * i}
        for i in range(num_points)
    ]
    all_scores = [score(**params) for params in all_params]

    for parameter_names in (["a", "b"], ["a", "c", "d"], ["a", "b", "c", "d"]):
        fig = plot_hyperparameter_results(
            all_params,
            all_scores,
            parameter_names=parameter_names,
            title=f"With {len(parameter_names)} variables",
            colorbar_label="Score",
        )
        save_figure(fig, f"test{len(parameter_names)}.png")
    fig = plot_hyperparameter_results(
        all_params,
        all_scores,
        parameter_names=["a", "b", "c", "d"],
        bounds=dict(a=(-50, 100)),
        title="With altered boundaries",
        colorbar_label="Score",
        colorbar_bounds=(-1000, 300),
    )
    save_figure(fig, "test-alt.png")
