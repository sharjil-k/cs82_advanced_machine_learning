import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from sklearn import model_selection
from sklearn.model_selection import learning_curve

def five_fold_cv_calculation():
    colours = ['red', 'blue', 'purple', 'green', 'orange']
    fontsize = 14
    fig7 = plt.figure(figsize=(12,6))
    plt.suptitle('5-fold Cross-Validation and metric calculation', fontsize=18)
    ax = fig7.add_subplot(111)
    ax.axis('off')
    ax.text(-0.05, 0.5, 'Iteration:',
            fontsize=fontsize,
            horizontalalignment='right',
            verticalalignment='center',
            rotation='vertical',
            transform=ax.transAxes)
    ax.annotate('Average all the metrics from each iteration', xy=(1.25, 0.5), xytext=(1.35, 0.5), xycoords='axes fraction', 
                fontsize=16, ha='right', va='center', rotation='vertical',
                bbox=dict(boxstyle='square', fc='white'),
                arrowprops=dict(arrowstyle='-[, widthB=10.0, lengthB=1.0', lw=2.0))
    for p in [
        patches.Rectangle( (0.20*(3-i-j), 0.20*j), 0.2, 0.16, hatch='.', fill=False, edgecolor=colours[j])
        for i in range(5)  for j in range(5)
    ]:
        ax.add_patch(p)
    for p in [
        patches.Rectangle( (0.20*(4-j), 0.20*j), 0.2, 0.16, hatch='x', alpha=0.6, edgecolor='blue')
        for j in range(5)
    ]:
        ax.add_patch(p)
    for p in [
        patches.Rectangle( (0.20*(5-j+i), 0.20*j), 0.2, 0.16, hatch='.', fill=False, edgecolor=colours[j])
        for i in range(5)  for j in range(5)
    ]:
        ax.add_patch(p)    
    for j in range(5): 
        ax.text(0.2*j+0.06, 1.0 , 'fold #' + str(j+1), fontsize=fontsize)
        ax.text(-0.03, 0.2*j+0.06, 5-j, fontsize=fontsize)
        ax.annotate('', fontsize=fontsize, xy=(1.05, 0.2*j+0.07), xycoords='axes fraction', xytext=(1.0, 0.2*j+0.07), 
                arrowprops=dict(arrowstyle="->"))
        ax.text(1.05, 0.2*j+0.06, 'calculate metric', fontsize=fontsize)

def format_axes(ax):
    fontsize = 12
    ax.margins(0.2)
    ax.set_xlabel('Factor 1', fontsize=fontsize)
    ax.set_ylabel('Factor 2', fontsize=fontsize)
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])
    ax.set_xlim((0,6))
    ax.set_ylim((0,6))

def plot_experimentation_strategies():

    marker_style = dict(color='cornflowerblue', linestyle=':', marker='o', markersize=15)
    fig, (ax1, ax2, ax3) = plt.subplots(figsize=(14, 4),nrows=1, ncols=3)
    ax1.set_title("(a) Best guess", fontsize=14)
    ax1.plot([2,3,4,4], [1,3,4,2], fillstyle='full', **marker_style)
    format_axes(ax1)
    ax2.set_title("(b) One by one", fontsize=14)
    ax2.plot([1,2] + [3]*7 + [4,5], [3]*3+list(range(1,6))+[3]*3, fillstyle='full', **marker_style)
    format_axes(ax2)
    ax3.set_title("(c) Grid design", fontsize=14)
    ax3.plot(np.repeat(range(1,6),5),np.tile(range(1,6),5), fillstyle='full', **marker_style)
    format_axes(ax3)

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.
    From the SK Learn documentation at
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : integer, cross-validation generator, optional
        If an integer is passed, it is the number of folds (defaults to 3).
        Specific cross-validation objects can be passed, see
        sklearn.cross_validation module for the list of possible objects

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt
