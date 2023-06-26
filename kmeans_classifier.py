import cv2
import numpy as np
from sklearn.cluster import KMeans


def find_closest_color(rgb_color):
    color_mapping = {
        (255, 255, 255): "white",
        (128, 62, 62): "red",
        (62, 62, 128): "blue",
        (62, 128, 62): "green",
        (240, 240, 15): "yellow",
        (1, 1, 1): "black",
    }
    min_distance = float("inf")
    closest_color = None

    for mapped_color, color_name in color_mapping.items():
        distance = np.linalg.norm(np.array(rgb_color) - np.array(mapped_color))
        if distance < min_distance:
            min_distance = distance
            closest_color = color_name

    return closest_color


def pixels_kmeans(image_path, colors_number):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    image = image.reshape(
        image.shape[0] * image.shape[1], 3
    )  # Reshape the image to a 2D array
    k = 3  # Number of clusters
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(image)
    dominant_colors = kmeans.cluster_centers_
    pixel_counts = np.bincount(
        kmeans.labels_
    )  # Count the number of pixels in each cluster
    sorted_colors = dominant_colors[np.argsort(pixel_counts)][::-1][:colors_number]
    named_sorted_colors = [find_closest_color(color) for color in sorted_colors]

    return named_sorted_colors


if __name__ == "__main__":
    image_path = "./crops/gavE7YPYlCEl6QpWM_9M5w.jpg.jpg"
    print(pixels_kmeans(image_path, 3))
