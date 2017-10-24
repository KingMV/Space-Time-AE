import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_loss(losses, valid_losses):
    """
    plot training loss vs. iteration number
    """
    plt.figure()
    plt.plot(range(len(losses)), losses, 'b', alpha=0.6, linewidth=0.5, label="training loss")
    valid_loss_every = (len(losses) - 1) / (len(valid_losses) - 1)
    plt.plot(range(0, len(losses), valid_loss_every), valid_losses, 'r', linewidth=0.5, label="validation loss")
    plt.xlabel("Iteration")
    plt.ylabel("Total loss")
    plt.legend(loc='upper right')
    plt.savefig("../results/Loss.png")


def plot_auc(aucs):
    """
    plot area under the curve vs. (iteration number / auc_every)
    """
    plt.figure()
    plt.plot(range(1, len(aucs) + 1), aucs)
    plt.xlabel("Training progress (# iter / constant)")
    plt.ylabel("Area under the roc curve")
    plt.savefig("../results/AUC.png")


def plot_regularity(regularity_scores, labels):
    """
    plot regularity score vs. frame number and shade anomalous background using ground truth labels
    """
    plt.figure()
    plt.plot(range(1, regularity_scores.shape[0] + 1), regularity_scores, linewidth=0.5)
    plt.xlabel("Frame number")
    plt.ylabel("Regularity score")
    for i in xrange(1, labels.shape[0] + 1):
        if labels[i - 1] == 1:
            plt.axvspan(i, i + 1, facecolor='salmon', alpha=0.5)
    plt.savefig("../results/Regularity.png")
