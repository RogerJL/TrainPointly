'''
point cloud data is stored as a 2D matrix
each row has 3 values i.e. the x, y, z value for a point

Project has to be submitted to github in the private folder assigned to you
Readme file should have the numerical values as described in each task
Create a folder to store the images as described in the tasks.

Try to create commits and version for each task.

'''
#%%
%matplotlib qt
import matplotlib
import numpy as np
from scipy.spatial import KDTree
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#%% utility functions
def show_cloud(points_plt):
    ax = plt.axes(projection='3d')
    ax.scatter(points_plt[:,0], points_plt[:,1], points_plt[:,2], s=0.01)
    plt.show()

def show_scatter(x,y):
    plt.scatter(x, y)
    plt.show()

def get_ground_level(pcd, plot=False):
    bins = 200
    h, limits = np.histogram(pcd[:,2], bins=bins, density=False)
    index_max = np.argmax(h)
    index = index_max
    while h[index + 1] < h[index]:  # decreasing
        index += 1
    print(index_max, limits[index_max + 1], index, limits[index + 1], h[:index + 10])

    if plot:
        width = np.diff(limits)  # calculate bar width
        plt.bar(limits[:-1] + 0.5 * width, height=h, width=width)

    return limits[index + 1]

#%% read file containing point cloud data
pcd1 = np.load("dataset1.npy")

pcd1.shape

#%% show downsampled data in external window

plt.figure("Dataset 1")
show_cloud(pcd1)

print(f"min {np.min(pcd1, axis=0)}, max {np.max(pcd1, axis=0)}")

if False:
    cut1 = 90
    cut = pcd1[np.logical_and(cut1 < pcd1[:, 1], pcd1[:, 1] < cut1 + 1)]
    show_scatter(cut[:,0], cut[:, 2])

    cut2 = 120
    cut = pcd1[np.logical_and(cut2 < pcd1[:, 1], pcd1[:, 1] < cut2 + 1)]
    show_scatter(cut[:,0], cut[:, 2])

    cut3 = 150
    cut = pcd1[np.logical_and(cut3 < pcd1[:, 1], pcd1[:, 1] < cut3 + 1)]
    show_scatter(cut[:,0], cut[:, 2])

#show_cloud(pcd[::10]) # keep every 10th point

#%% remove ground plane

'''
Task 1 (3)
find the best value for the ground level
One way to do it is useing a histogram 
np.histogram

update the function get_ground_level() with your changes

For both the datasets
Report the ground level in the readme file in your github project
Add the histogram plots to your project readme
'''
plt.figure("Dataset 1 - Histograms of height")
est_ground_level = get_ground_level(pcd1, plot=True)  # 61.83

print(est_ground_level)

pcd1_above_ground = pcd1[pcd1[:, 2] > est_ground_level]
#%%
pcd1_above_ground.shape

#%% side view
plt.figure("Dataset 1 - removed ground")
show_cloud(pcd1_above_ground)

#%%
pcd2 = np.load("dataset2.npy")
plt.figure("Dataset 2")
show_cloud(pcd2)
print(f"min {np.min(pcd2, axis=0)}, max {np.max(pcd2, axis=0)}")
#%%
plt.figure("Dataset 2 - Histograms of height")
est2_ground_level = get_ground_level(pcd2, plot=True)
print(est2_ground_level)  # 61.97

pcd2_above_ground = pcd2[pcd2[:,2] > est2_ground_level]
plt.figure("Dataset 2 - removed ground")
show_cloud(pcd2_above_ground)

# %%
bad_eps = 10
# find the elbow
clustering_bad = DBSCAN(eps = bad_eps, min_samples=7).fit(pcd1_above_ground)

#%%
clusters = len(set(clustering_bad.labels_)) - (1 if -1 in clustering_bad.labels_ else 0)
colors = [plt.cm.jet(each) for each in np.linspace(0, 1, clusters)]

# %%
# Plotting resulting clusters
plt.figure("clusters-bad", figsize=(10,10))
plt.scatter(pcd1_above_ground[:, 0],
            pcd1_above_ground[:, 1],
            c=clustering_bad.labels_,
            cmap=matplotlib.colors.ListedColormap(colors),
            s=2)


plt.title(f'DBSCAN: {clusters} clusters eps={bad_eps}, {np.sum(clustering_bad.labels_ == -1)} noice', fontsize=20)
plt.xlabel('x axis', fontsize=14)
plt.ylabel('y axis', fontsize=14)
plt.show()

#%%
'''
Task 2 (+1)

Find an optimized value for eps.
Plot the elbow and extract the optimal value from the plot
Apply DBSCAN again with the new eps value and confirm visually that clusters are proper

https://www.analyticsvidhya.com/blog/2020/09/how-dbscan-clustering-works/
https://machinelearningknowledge.ai/tutorial-for-dbscan-clustering-in-python-sklearn/

For both the datasets
Report the optimal value of eps in the Readme to your github project
Add the elbow plots to your github project Readme
Add the cluster plots to your github project Readme
'''

#%%
# https://www.analyticsvidhya.com/blog/2020/09/how-dbscan-clustering-works/
# https://machinelearningknowledge.ai/tutorial-for-dbscan-clustering-in-python-sklearn/

from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator

def determine_elbow(pcd_above_ground_, id="", plot=False):

    neigh = NearestNeighbors(n_neighbors=2)
    nbrs = neigh.fit(pcd_above_ground_)
    distances, indices = nbrs.kneighbors(pcd_above_ground_)
    distances = np.sort(distances, axis=0)
    distances = distances[:, 1]

    i = np.arange(len(distances))
    # polynomial gives knees before joint, not great in this case
    knee = KneeLocator(i, distances, S=1, curve='convex', direction='increasing', interp_method='interp1d')

    if plot:
        #fig = plt.figure("elbow" + str(id), figsize=(5, 5))
        knee.plot_knee(title="Elbow plot " + str(id), xlabel="Points", ylabel="Distance")
        plt.show()

    return distances[knee.knee]

eps1 = determine_elbow(pcd1_above_ground, id=1, plot=True)
eps2 = determine_elbow(pcd2_above_ground, id=2, plot=True)
# %%
min_samples = 2
# find the elbow
clustering1 = DBSCAN(eps=eps1, min_samples=min_samples).fit(pcd1_above_ground)
clustering2 = DBSCAN(eps=eps2, min_samples=min_samples).fit(pcd2_above_ground)

#%%
def show_clustering(pcd_above_ground, clustering, eps=-1.0, min_neighbours="unknown", id=""):
    clusters = len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)
    colors = [plt.cm.jet(each) for each in np.linspace(0, 1, clusters)]
    # Plotting resulting clusters
    plt.figure("clusters" + str(id), figsize=(10, 10))
    plt.scatter(pcd_above_ground[:, 0],
                pcd_above_ground[:, 1],
                c=clustering.labels_,
                cmap=matplotlib.colors.ListedColormap(colors),
                s=1.5)

    plt.title(f'DBSCAN({eps:.3f}, {min_neighbours}): {clusters} clusters, {np.sum(clustering.labels_ == -1)} noice',
              fontsize=20)
    plt.xlabel('x axis', fontsize=14)
    plt.ylabel('y axis', fontsize=14)
    plt.show()

show_clustering(pcd1_above_ground, clustering1, eps=eps1, min_neighbours=min_samples, id=1)
show_clustering(pcd2_above_ground, clustering2, eps=eps2, min_neighbours=min_samples, id=2)

# %%


#%%
'''
Task 3 (+1)

Find the largest cluster, since that should be the catenary, 
beware of the noise cluster.

Use the x,y span for the clusters to find the largest cluster

For both the datasets
Report min(x), min(y), max(x), max(y) for the catenary cluster in the Readme of your github project
Add the plot of the catenary cluster to the readme

'''
