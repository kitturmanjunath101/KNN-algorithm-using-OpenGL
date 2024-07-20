import glfw
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import numpy as np
from collections import Counter

# Initialize the library
if not glfw.init():
    raise Exception("GLFW can't be initialized")

# Create a windowed mode window and its OpenGL context
window = glfw.create_window(800, 600, "KNN Visualization with User Input", None, None)
if not window:
    glfw.terminate()
    raise Exception("GLFW window can't be created")

# Make the window's context current
glfw.make_context_current(window)

# Set the viewport size
glViewport(0, 0, 800, 600)
glMatrixMode(GL_PROJECTION)
glLoadIdentity()
gluOrtho2D(0, 800, 0, 600)
glMatrixMode(GL_MODELVIEW)
glLoadIdentity()

class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predictions = [self._predict(x) for x in X_test]
        return np.array(predictions)

    def _predict(self, x):
        distances = [self._euclidean_distance(x, x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

    def _euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

# Example data for 3 clusters
X_train = np.array([
    [100, 150], [150, 200], [200, 250],  # Cluster 1
    [400, 450], [450, 500], [500, 550],  # Cluster 2
    [100, 450], [150, 500], [200, 550]   # Cluster 3
])
y_train = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])

# Create KNN classifier
knn = KNN(k=3)
knn.fit(X_train, y_train)

X_test = []

def draw_point(x, y, color):
    glColor3f(*color)
    glBegin(GL_QUADS)
    glVertex2f(x - 5, y - 5)
    glVertex2f(x + 5, y - 5)
    glVertex2f(x + 5, y + 5)
    glVertex2f(x - 5, y + 5)
    glEnd()

# Flags to ensure the print statements are executed only once per cluster
printed_cluster_1 = False
printed_cluster_2 = False
printed_cluster_3 = False

def draw_knn():
    global printed_cluster_3, printed_cluster_2, printed_cluster_1
    glClear(GL_COLOR_BUFFER_BIT)

    # Draw training points
    for point, label in zip(X_train, y_train):
        if label == 0:
            color = (1.0, 0.0, 0.0)  # Red for cluster 1
        elif label == 1:
            color = (0.0, 1.0, 0.0)  # Green for cluster 2
        else:
            color = (0.0, 0.0, 1.0)  # Blue for cluster 3
        draw_point(point[0], point[1], color)

    # Draw test points with predictions
    if X_test:
        predictions = knn.predict(X_test)
        for point, prediction in zip(X_test, predictions):
            if prediction == 0:
                if not printed_cluster_1:
                    print("The given test points belong to cluster 1 (Red cluster)")
                    printed_cluster_1 = True
                color = (1.0, 1.0, 0.0)  # Yellow for cluster 1
            elif prediction == 1:
                if not printed_cluster_2:
                    print("The given test points belong to cluster 2 ( Green cluster)")
                    printed_cluster_2 = True
                color = (0.0, 1.0, 1.0)  # Cyan for cluster 2
            else:
                if not printed_cluster_3:
                    print("The given test points belong to cluster 3 (Magenta cluster)")
                    printed_cluster_3 = True
                color = (1.0, 0.0, 1.0)  # Magenta for cluster 3
            draw_point(point[0], point[1], color)

    glfw.swap_buffers(window)
def get_user_input():
    try:
        print("Enter a random dataset")
        x = int(input("Enter x coordinate of the test point (0-800): "))
        y = int(input("Enter y coordinate of the test point (0-600): "))
        if 0 <= x <= 800 and 0 <= y <= 600:
            X_test.append([x, 600 - y])  # Convert to OpenGL coordinates
            draw_knn()
        else:
            print("Coordinates out of bounds. Please enter values within the range.")
    except ValueError:
        print("Invalid input. Please enter integer values.")

# Main loop
while not glfw.window_should_close(window):
    draw_knn()
    glfw.poll_events()
    if glfw.get_key(window, glfw.KEY_K) == glfw.PRESS:
        get_user_input()

glfw.terminate()
