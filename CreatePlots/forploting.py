import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

# Generate some random points in 2D space
np.random.seed(100)
points = np.random.rand(20, 2)

# Specify the value of k for k-nearest neighbors
k_neighbors = 4

# Find the k-nearest neighbors for each point
nbrs = NearestNeighbors(n_neighbors=k_neighbors, algorithm='ball_tree').fit(points)
distances, indices = nbrs.kneighbors(points)

plt.subplot(121)

w=0.8
# Plot arrows to k-nearest neighbors
for i in range(len(points)):
    j=indices[i][k_neighbors-1]
    if i==0:
        plt.arrow(points[i, 0], points[i, 1], (points[j, 0]*w - points[i, 0]*w), points[j, 1]*w - points[i, 1]*w,
                  head_width=0.02, head_length=0.02, fc='black', ec='black',label="3rd nearest neighbor")
    else:
        plt.arrow(points[i, 0], points[i, 1], (points[j, 0] * w - points[i, 0] * w),
                  points[j, 1] * w - points[i, 1] * w,
                  head_width=0.02, head_length=0.02, fc='black', ec='black')




# Plot the points
plt.scatter(points[:, 0], points[:, 1], color='blue',label="Data samples",s=60)

# Place text labels above each point
for i, (x, y) in enumerate(points):
    plt.text(x, y + 0.03, f't{i}', ha='center', va='bottom', color='black')


# Set labels and show the plot
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Points with Arrows to 3-nearest Neighbors')
plt.legend()

plt.subplot(122)

plt.plot(distances[:,k_neighbors-1],linewidth=3)
plt.xticks(range(len(points)),[f"t{t}" for t in range(len(points))])

plt.show()