"""Uses matplotlib to handle the visualization of the results from searching 
methods. Note that this file cannot be loaded if ROOT is also loaded in because
matplotlib and ROOT do not work together."""

from copy import copy
import itertools
import numpy as np
import math

import matplotlib

matplotlib.use("agg")
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec


__all__ = ["plotSearchResults"]


def _get_max_scores(param1, param2, searched_points):
    """Determines the maximum score for each combination of parameter values

    :param param1: string for the first parameter name
    :param param2: string for the second parameter name
    :param searched_points: the list of all points which were searched
    :returns: numpy arrays for p1 values, p2 values, and corresponding scores
    """
    p1 = list()
    p2 = list()
    scores = list()
    for params, score in searched_points:
        param1_val = params[param1]
        param2_val = params[param2]

        found = False
        for i, (p1_val, p2_val) in enumerate(zip(p1, p2)):
            if p1_val == param1_val and p2_val == param2_val:
                found = True
                scores[i] = max(score, scores[i])
                break
        if not found:
            p1.append(param1_val)
            p2.append(param2_val)
            scores.append(score)
    return np.array(p1), np.array(p2), np.array(scores)


def plotSearchResults(
    searchResults,
    filename,
    plot_title,
    directory=".",
    diagnostics=0,
    bounds=None,
):
    """Plots the results that a Searcher object determined

    :param searchResults: (max_idx, all_methods, all_params, all_scores)
    :param filename: File name within the directory (not including .png)
    :param plot_title: Title to put at the top of the plots
    :param directory: Optional directory to save the file
    :param diagnostics: 0 for no printouts, 1 for some printouts, 2 for all
        printouts. defaults to 0
    :param bounds: possibly a 2-tuple (min, max) for the bounds to use to
        normalize the colorbars for the scores
    """

    max_idx, all_methods, all_params, all_scores = searchResults

    if bounds is None:
        bounds = (None, None)

    searched_points = list(zip(all_params, all_scores))
    parameter_names = list(all_params[max_idx].keys())
    parameter_names.sort()

    best_params, best_score = searched_points[max_idx]

    limits = dict()
    for key in parameter_names:
        values = [x[key] for x in all_params]
        min_val = min(values)
        max_val = max(values)
        interval = max_val - min_val
        if interval == 0:
            if diagnostics:
                print(
                    "WARNING: Range of tested values for key %s is 0. Widening..."
                    % key
                )
            limits[key] = (min_val - 0.5, max_val + 0.5)
        else:
            limits[key] = (min_val - 0.1 * interval, max_val + 0.1 * interval)

    # Determine how to position all of the plots on the page
    num_params = len(parameter_names)
    if num_params == 1:
        num_figs = 1
    else:
        num_figs = int(
            math.factorial(num_params) / (2 * math.factorial(num_params - 2))
        )  # binomial coefficient
    num_figs_per_side = int(math.ceil(math.sqrt(num_figs)))
    # Actually make the figure, ensuring enough space for each plot (and the words around them (+2))
    fig = plt.figure(figsize=(6 * num_figs_per_side + 1, 6 * num_figs_per_side))
    gs = GridSpec(5 * num_figs_per_side, 5 * num_figs_per_side)
    counter = 0
    # iterate over each unique combination of parameters
    axs = []
    for c1, param1 in enumerate(parameter_names):
        for c2, param2 in enumerate(parameter_names):
            if c2 <= c1 and num_params != 1:
                continue  # ensure we don't get repeated plots

            region_x, region_y = (
                counter % num_figs_per_side,
                counter // num_figs_per_side,
            )
            counter += 1
            # grab the region for this plot
            ax = plt.subplot(
                gs[
                    5 * region_y : 5 * (region_y + 1),
                    5 * region_x : 5 * (region_x + 1),
                ]
            )

            # determine the maximum scores for these parameters
            p1_vals, p2_vals, scores = _get_max_scores(
                param1, param2, searched_points
            )

            ax.set_facecolor("grey")
            try:
                if bounds[0] is not None and bounds[1] is not None:
                    levels = np.linspace(bounds[0], bounds[1], num=21)
                else:
                    levels = None
                cbar = ax.tricontourf(
                    p2_vals,
                    p1_vals,
                    scores,
                    cmap=plt.get_cmap("hot"),
                    vmin=bounds[0],
                    vmax=bounds[1],
                    levels=levels,
                )
            except:
                # If this happens, we probably have "singular" plotting data, so use a scatter plot
                cbar = ax.scatter(
                    p2_vals,
                    p1_vals,
                    c=scores,
                    cmap=plt.get_cmap("hot"),
                    marker="o",
                    vmin=bounds[0],
                    vmax=bounds[1],
                )
            ax.axvline(best_params[param2], color="blue", linestyle="--")
            ax.axhline(best_params[param1], color="blue", linestyle="--")
            ax.plot(best_params[param2], best_params[param1], "bo")
            ax.set_xlabel(param2)
            ax.set_ylabel(param1)
            ax.set_xlim(*limits[param2])
            ax.set_ylim(*limits[param1])
            ax.set_title("%s and %s" % (param1, param2))
            axs += [ax]
    # # set single colorbar
    # # fig.subplots_adjust(right=0.8)
    # ax = plt.subplot(gs[:, -1])
    # plt.colorbar(cbar, cax=ax)

    supertitle = plt.suptitle("%s\nBest Score: %f" % (plot_title, best_score))

    # There is a chance that the tight_layout will fail, so we make a "rough" plot here just in case
    filepath = "%s/Search_%s_PRE.png" % (directory, filename)
    print("Saving pre-file to %s" % filepath)
    plt.savefig(filepath, bbox_inches="tight", pad_inches=0.5)

    # The GridSpec needs to do this instead, because it does it better, rect accounts for the super title
    gs.tight_layout(fig, rect=[0, 0.08, 1, 0.90], h_pad=0.5)

    # Shift the title to the right spot
    supertitle.set_y(0.95)

    cax = fig.add_axes([0.09, 0.06, 0.84, 0.02])
    plt.colorbar(cbar, cax=cax, orientation="horizontal")
    cax.set_xlabel("Score")

    # save the plot
    filepath = "%s/Search_%s.png" % (directory, filename)
    print("Saving file to %s" % filepath)
    plt.savefig(filepath, bbox_inches="tight", pad_inches=0.5)


# Unit test
if __name__ == "__main__":

    def score(**params):
        return sum(params.values())

    max_idx = 1

    num_points = 3 * 3 * 2 * 2

    all_methods = ["method_%i" % i for i in range(num_points)]
    all_params = [
        {"a": i, "b": 3 - 2 * i, "c": i * i, "d": 3 - 2 * i * i}
        for i in range(num_points)
    ]
    all_scores = [score(**params) for params in all_params]
    max_idx = np.argmax(all_scores)

    searchResults = (max_idx, all_methods, all_params, all_scores)

    plotSearchResults(searchResults, "test", "Test Plot", diagnostics=2)

