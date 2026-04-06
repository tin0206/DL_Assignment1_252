import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datasets import load_dataset
from sklearn.metrics.pairwise import cosine_similarity

os.makedirs("results", exist_ok=True)

print(">>> Loading CIFAR-10 dataset...")
dataset = load_dataset("uoft-cs/cifar10")
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


def plot_distribution(split_name, title):
    labels = list(dataset[split_name]['label'])

    df = pd.DataFrame(labels, columns=['label'])
    counts = df['label'].value_counts().sort_index()

    plt.figure(figsize=(12, 6))
    ax = sns.barplot(x=[classes[i] for i in counts.index], y=counts.values, palette='magma')

    plt.title(title, fontsize=15)
    plt.xlabel('Class Name', fontsize=12)
    plt.ylabel('Number of Samples', fontsize=12)
    plt.ylim(0, max(counts.values) * 1.1)

    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 10), textcoords='offset points')

    plt.savefig(f"{split_name}_distribution.png")
    plt.show()


def plot_statistical_color_profile(dataset):
    images = np.array(dataset['img'])

    colors = ('red', 'green', 'blue')
    plt.figure(figsize=(10, 6))

    for i, color in enumerate(colors):
        hist, bin_edges = np.histogram(images[:, :, :, i], bins=256, range=(0, 255))

        plt.plot(bin_edges[0:-1], hist, color=color, label=f'{color.capitalize()} channel', linewidth=1.5)

        channel_mean = np.mean(images[:, :, :, i])
        print(f"{color.capitalize()} Channel Mean: {channel_mean:.2f}")

    plt.title("Figure 3: CIFAR-10 Raw Pixel Intensity Distribution", fontsize=12)
    plt.xlabel("Pixel Value (0-255)")
    plt.ylabel("Frequency (Number of Pixels)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.savefig("results/pixel_distribution.png", dpi=300, bbox_inches='tight')
    plt.show()


def plot_semantic_class_similarity(dataset, classes):
    class_means = []

    print("Calculating average feature vectors per class...")
    for i in range(10):
        class_indices = [idx for idx, label in enumerate(dataset['label']) if label == i]
        class_images = np.array(dataset['img'])[class_indices]

        flattened_images = class_images.reshape(len(class_images), -1)

        class_means.append(np.mean(flattened_images, axis=0))

    similarity_matrix = cosine_similarity(class_means)

    plt.figure(figsize=(12, 10))
    sns.heatmap(similarity_matrix, annot=True, fmt='.3f', cmap='magma',
                xticklabels=classes, yticklabels=classes,
                cbar_kws={'label': 'Cosine Similarity Score'})

    plt.title("Figure 4: Heatmap of Semantic Similarity Between CIFAR-10 Classes", fontsize=14)
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig("results/class_similarity_heatmap.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    plot_distribution('train', 'Class Distribution: CIFAR-10 Training Set')

    plot_statistical_color_profile(dataset['train'])

    plot_semantic_class_similarity(dataset['train'], classes)

    print(">>> EDA Complete! Check the 'results/' folder for images.")
