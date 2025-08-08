import sys
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

# --- Utility functions ---

def show_cloud(points_plt, title=None, save_path=None):
    """
    Displays a 3D scatter plot of point cloud data.
    Optionally saves the plot to a file.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points_plt[:,0], points_plt[:,1], points_plt[:,2], s=0.01)
    if title:
        plt.title(title)
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.close()

def get_ground_level(z_values, dataset_name):
    """
    Determines the ground level using a histogram of Z-values.
    Saves the histogram plot with the detected ground level marked.
    """
    # Calculate histogram of Z-values
    hist, bin_edges = np.histogram(z_values, bins=100)
    ground_index = np.argmax(hist)  # index of the peak frequency
    ground_level = (bin_edges[ground_index] + bin_edges[ground_index + 1]) / 2

    # Plot and save histogram
    plt.figure()
    plt.hist(z_values, bins=100)
    plt.axvline(ground_level, color='red', linestyle='--', label=f"Ground Level: {ground_level:.2f}")
    plt.legend()
    plt.title(f"Ground Level Histogram - {dataset_name}")
    plt.xlabel("Z value")
    plt.ylabel("Frequency")
    plt.savefig(f"plots/{dataset_name}_hist.png", dpi=300)
    plt.close()

    return ground_level

def find_optimal_eps(X, dataset_name, k=4):
    """
    Estimates the optimal eps value for DBSCAN using the k-distance (elbow) method.
    Saves the elbow plot.
    """
    # Find nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=k).fit(X)
    distances, _ = nbrs.kneighbors(X)
    distances = np.sort(distances[:, k-1])

    # Plot and save elbow graph
    plt.figure()
    plt.plot(distances)
    plt.ylabel("k-distance")
    plt.xlabel("Points sorted by distance")
    plt.title(f"Elbow plot for eps - {dataset_name}")
    plt.savefig(f"plots/{dataset_name}_elbow.png", dpi=300)
    plt.close()

    # Use the 75th percentile as a simple heuristic for eps
    eps_guess = np.percentile(distances, 75)
    return eps_guess

def cluster_and_plot(X, eps, dataset_name):
    """
    Runs DBSCAN clustering on the given data and saves a cluster plot.
    """
    db = DBSCAN(eps=eps, min_samples=5).fit(X)
    labels = db.labels_
    clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"[{dataset_name}] Found {clusters} clusters with eps={eps:.3f}")

    # Plot and save clusters
    plt.figure(figsize=(8,8))
    plt.scatter(X[:,0], X[:,1], c=labels, cmap='tab20', s=1)
    plt.title(f"DBSCAN Clusters - {dataset_name} (eps={eps:.3f})")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.savefig(f"plots/{dataset_name}_clusters.png", dpi=300)
    plt.close()

    return labels

def find_largest_cluster(X, labels, dataset_name):
    """
    Identifies the largest non-noise cluster and calculates its bounding box.
    Saves the catenary cluster plot.
    """
    # Remove noise label (-1)
    unique_labels = set(labels)
    unique_labels.discard(-1)

    # Find cluster with the most points
    largest_label = max(unique_labels, key=lambda l: np.sum(labels == l))
    cluster_points = X[labels == largest_label]

    # Calculate bounding box
    min_x, min_y = cluster_points[:,0].min(), cluster_points[:,1].min()
    max_x, max_y = cluster_points[:,0].max(), cluster_points[:,1].max()

    # Plot and save largest cluster
    plt.figure(figsize=(8,8))
    plt.scatter(cluster_points[:,0], cluster_points[:,1], s=1)
    plt.title(f"Catenary Cluster - {dataset_name}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.savefig(f"plots/{dataset_name}_catenary.png", dpi=300)
    plt.close()

    return (min_x, min_y, max_x, max_y)

# --- Main script execution ---
if __name__ == "__main__":
    # Check if dataset path is provided
    if len(sys.argv) < 2:
        print("Usage: python assignment5.py <dataset.npy>")
        sys.exit(1)

    dataset_path = sys.argv[1]
    dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]

    # Ensure plots folder exists
    os.makedirs("plots", exist_ok=True)

    # Load dataset
    pcd = np.load(dataset_path)
    print(f"[{dataset_name}] Loaded point cloud shape: {pcd.shape}")

    # --- Task 1: Ground level detection ---
    ground_level = get_ground_level(pcd[:,2], dataset_name)
    print(f"[{dataset_name}] Ground level: {ground_level:.3f}")

    # Filter out ground points
    pcd_above_ground = pcd[pcd[:,2] > ground_level]

    # --- Task 2: Optimal eps estimation ---
    eps_value = find_optimal_eps(pcd_above_ground[:,:2], dataset_name)
    print(f"[{dataset_name}] Estimated optimal eps: {eps_value:.3f}")

    # Run clustering and plot results
    labels = cluster_and_plot(pcd_above_ground[:,:2], eps_value, dataset_name)

    # --- Task 3: Largest cluster detection ---
    bounds = find_largest_cluster(pcd_above_ground[:,:2], labels, dataset_name)
    print(f"[{dataset_name}] Catenary cluster bounds: min_x={bounds[0]:.3f}, min_y={bounds[1]:.3f}, max_x={bounds[2]:.3f}, max_y={bounds[3]:.3f}")

    print(f"[{dataset_name}] All plots saved in 'plots/' folder.")
