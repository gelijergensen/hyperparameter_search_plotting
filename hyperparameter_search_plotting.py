import os.path

import matplotlib as mpl

mpl.use("agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def plot_datapoints(ax, xvals, yvals, scores, background_color="grey"):
    ax.set_facecolor(background_color)
    try:
        ax.tricontourf(xvals, yvals, scores)
    except:
        # If something goes wrong, we revert to a scatter plot
        ax.scatter(xvals, yvals, c=scores)
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


# TODO:
# decide on a form for the inputs
# - dictionaries of lists? list of lists? parameter names on the side?
# function to yield the axes
# method for cleanly plotting all things for the single axis
# - try to decouple from form of the inputs
# configuration
# - objects? long set of kwargs?
# handle normalization of color in tricolorf!

if __name__ == "__main__":

    xvalues = [0, 1, 2, 3, 4]
    yvalues = [0, 1, 2, 3, 4]
    scores = [0, 1, 2, 3, 4]
    cmap = plt.get_cmap("hot")

    fig = plt.figure()
    gs = GridSpec(2, 2, figure=fig)
    ax = fig.add_subplot(gs[0, 0])
    ax.scatter(xvalues, yvalues, c=scores, cmap=cmap)
    # ax.set_title("I have a title!")
    ax = fig.add_subplot(gs[1, 0])
    ax.scatter(xvalues, yvalues[::-1], c=scores, cmap=cmap)
    ax = fig.add_subplot(gs[0, 1])
    ax.scatter(xvalues[::-1], yvalues, c=scores, cmap=cmap)
    ax = fig.add_subplot(gs[1, 1])
    ax.scatter(xvalues[::-1], yvalues[::-1], c=scores, cmap=cmap)
    ax.set_title("This one also has a title")

    fig = finalize_figure(fig, gs, supertitle="Supertitle!")
    save_figure(fig, "test.png")
