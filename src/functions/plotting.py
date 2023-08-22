import matplotlib.pyplot as plt
import numpy as np

from scipy.stats import norm
from matplotlib.ticker import MaxNLocator


def plot_single_output_history(hist, outlier_threshold=None, to_file: str = None) -> None:
    train_loss = np.array(hist['loss'])
    val_loss = np.array(hist['val_loss'])

    if outlier_threshold is None:
        Q1 = np.percentile(train_loss, 25)
        Q3 = np.percentile(train_loss, 75)
        IQR = Q3 - Q1
        outlier_threshold = Q3 + 1.5 * IQR

    train_loss_outliers = train_loss > outlier_threshold
    val_loss_outliers = val_loss > outlier_threshold
    train_loss_line = np.where(train_loss_outliers, np.nan, train_loss)
    val_loss_line = np.where(val_loss_outliers, np.nan, val_loss)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss_line, label='Train Loss')
    plt.plot(val_loss_line, label='Validation Loss')

    plt.plot(np.where(train_loss_outliers)[0], train_loss[train_loss_outliers], 'ro', label='Outliers')

    for i, loss in zip(np.where(train_loss_outliers)[0], train_loss[train_loss_outliers]):
        plt.text(i, loss, f'{loss:.2f}', color='red')

    plt.title('Loss Evolution')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(hist['accuracy'], label='Train accuracy')
    plt.plot(hist['val_accuracy'], label='Validation accuracy')
    plt.title('Metric Evolution')
    plt.xlabel('Epoch')
    plt.ylabel('Metric')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()

    if to_file is not None:
        plt.savefig(to_file, bbox_inches='tight')

    plt.show()


def plot_multi_output_history(
        hist,
        loss_key='loss',
        val_loss_key='val_loss',
        metric_key='ciou_metric',
        val_metric_key='val_ciou_metric') -> None:
    # Loss
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(hist.history1[loss_key], label='Train Loss')
    plt.plot(hist.history1[val_loss_key], label='Validation Loss')
    plt.title('Loss Evolution')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    # CIoU vals
    plt.subplot(1, 2, 2)
    plt.plot(hist.history1[metric_key], label='Train CIoU metric')
    plt.plot(hist.history1[val_metric_key], label='Validation CIoU metric')
    plt.title('Metric Evolution')
    plt.xlabel('Epoch')
    plt.ylabel('Metric')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_histograms(data: list[float], title: str, x_label: str, to_file: str = None) -> None:
    lim_from, lim_to = .7, 1
    bins = np.arange(0.85, 1.01, 0.01)
    counts, bin_edges = np.histogram(data, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    mu, std = np.mean(data), np.std(data)
    x = np.linspace(lim_from, lim_to, 1000)
    y = norm.pdf(x, mu, std) * len(data) * (bins[1] - bins[0])
    fig, ax = plt.subplots(figsize=(9, 6))

    ax.bar(bin_centers, counts, width=0.01, alpha=0.5, align='center', edgecolor='black', label=f'{x_label} Frequencies')
    ax.plot(x, y, 'r-', lw=2, label='Gaussian Distribution')
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlim(lim_from, lim_to)

    if to_file is not None:
        plt.savefig(to_file, bbox_inches='tight')

    plt.show()


def _rescale_to_0_255(img: np.ndarray) -> np.ndarray:
    img_min, img_max = img.min(), img.max()

    return (255 * (img - img_min) / (img_max - img_min)).astype(np.uint8)


def plot_images(original, centered):
    n = original.shape[0]

    plt.figure(figsize=(10, 5 * n))

    for i in range(n):
        plt.subplot(n, 2, i*2 + 1)
        plt.imshow(np.squeeze(original[i].astype('uint8')))
        plt.title('Original Image')
        plt.axis('off')

        rescaled_centered = _rescale_to_0_255(centered[i].numpy())

        plt.subplot(n, 2, i*2 + 2)
        plt.imshow(np.squeeze(rescaled_centered))
        plt.title('Augmented Image')
        plt.axis('off')

    plt.tight_layout()
    plt.show()
