'''
This is the .py file containing the relevant functions of the GPU accelerated DBSCAN 
algorithm and the visualisation of the clustering algorithm.
'''


import copy
import numpy as np

# For accelarated clustering on GPU
from cuml.cluster import DBSCAN
import cupy as cp
from time import time

# Plotting (of clusters)
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import mplhep as hep
plt.rcdefaults()
plt.style.use([hep.style.ROOT])
from skimage.measure import find_contours


def clustering_single_image(image, eps=1.5, min_samples=2, threshold=200, visualize=False):
    '''
    Function for finding the cluster as well as the number of clusters for one image.
    '''

    image = image.squeeze()

    # get the x and y coordinates of the pixels above threshold
    x_coords, y_coords = np.nonzero(image > threshold)

    # get all pixels as array of x and y coordinates
    points = np.column_stack((x_coords, y_coords))

    if points.shape[0] == 0:  # Handle case where all points are filtered out
        return 0, None, None, None

    # Move data to GPU for acceleration
    points_gpu = cp.asarray(points)

    # Run DBSCAN on the image
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean").fit(points_gpu)
    labels = dbscan.labels_ # get the labels
    
    # Count unique clusters (ignoring noise labeled -1)
    unique_clusters = cp.unique(labels)
    num_clusters = len(unique_clusters[unique_clusters != -1])

    # Visualization: Get a cluster map
    if visualize:

        # move cluster labels back to cpu
        labels_host = cp.asnumpy(labels) 

        # flatten the original image for the clustermap
        image_flat = image.reshape(-1,)
        mask = (image_flat > threshold)

        cluster_map = np.full_like(image_flat, fill_value=-1) # fill pixels that don't belong to a cluster with -1 to be ignored
        cluster_map[mask] = labels_host

        return num_clusters, cluster_map.reshape(16,8)

    return num_clusters, None


def fix_contours(contour):

    ''' 
    Function for finding the correct contours using brute force. 
    This is necessary since the find_contours function only find the points of the contour resulting in ugly 
    contour lines going through the pixels.
    '''

    y1, x1 = contour[0,:]
    new_contour = contour.copy()
    running_idx = 1
    flag1 = False
    flag2 = False
    flag3 = False
    flag4 = False


    for idx, (y, x) in enumerate(contour[1:,:]):
        

        if (((y-y1) == -0.5) and ((x-x1) == 0.5)):
            if flag1:
                new_contour = np.insert(new_contour, idx+running_idx, np.array([y1,x]), axis=0)
                flag1 = False
            else:
                new_contour = np.insert(new_contour, obj = idx+running_idx, values = np.array([y,x1]), axis=0)
                flag1 = True
            flag2 = False
            flag3 = False
            flag4 = False
            running_idx += 1

        elif (((y-y1) == -0.5) and ((x-x1) == -0.5)):
            if flag2:
                new_contour = np.insert(new_contour, obj = idx+running_idx, values = np.array([y,x1]), axis=0)
                flag2 = False
            else:
                new_contour = np.insert(new_contour, idx+running_idx, np.array([y1,x]), axis=0)
                flag2 = True
            flag1 = False
            flag3 = False
            flag4 = False
            running_idx += 1

        elif (((y-y1) == +0.5) and ((x-x1) == +0.5)):

            if flag3:
                new_contour = np.insert(new_contour, idx+running_idx, np.array([y,x1]), axis=0)
                flag3 = False
            else:
                new_contour = np.insert(new_contour, idx+running_idx, np.array([y1,x]), axis=0)
                flag3 = True
            flag1 = False
            flag2 = False
            flag4 = False
            running_idx += 1

        elif (((y-y1) == 0.5) and ((x-x1) == -0.5)):

            if flag4:
                new_contour = np.insert(new_contour, idx+running_idx, np.array([y1,x]), axis=0)
                flag4 = False
            else:
                new_contour = np.insert(new_contour, idx+running_idx, np.array([y,x1]), axis=0)
                flag4 = True
            flag1 = False
            flag2 = False
            flag3 = False
            running_idx += 1

        elif ((y-y1) == 1.5) and ((x-x1) == 1.5):
            new_contour = np.insert(new_contour,idx + running_idx, np.array([y1,x1 + 0.5], [y1 + 1, x1 + 0.5], [y1 + 1, x1 + 1.5]), axis = 0)
            running_idx += 3

        elif ((y-y1) == 1.5) and ((x-x1) == -1.5):
            new_contour = np.insert(new_contour,idx + running_idx, np.array([y1 + 0.5,x1], [y1 + 0.5, x1 - 1], [y1 + 1.5, x1 - 1]), axis = 0)
            running_idx += 3

        elif ((y-y1) == -1.5) and ((x-x1) == -1.5):
            new_contour = np.insert(new_contour,idx + running_idx, np.array([y1,x1 - 0.5], [y1 - 1, x1 - 0.5], [y1 -1, x1 - 1.5]), axis = 0)
            running_idx += 3

        elif ((y-y1) == -1.5) and ((x-x1) == 1.5):
            new_contour = np.insert(new_contour,idx + running_idx, np.array([y1 - 0.5,x1], [y1 - 0.5, x1 + 1], [y1 - 1.5, x1 + 1]), axis = 0)
            running_idx += 3

        x1, y1 = x, y

    return new_contour

    

def plot_clusters(image, cluster_map, num, title="Clusters in Calorimeter Image"):

    ''' Function for plotting showers with clusters marked red. '''

    fig = plt.figure(figsize=(6, 8))

    norm = mcolors.LogNorm(vmin=1, vmax=100e3)
    cmap = copy.copy(matplotlib.colormaps["viridis"])

    mask = (cluster_map == -1).astype(np.float32)
    cmap_mask = mcolors.ListedColormap([(0, 0, 0, 0), (0, 0, 0, 1)])  # Transparent und Schwarz

    im = plt.imshow(image, cmap=cmap, norm=norm, interpolation='nearest')
    # mask_image = plt.imshow(mask, cmap = cmap_mask, alpha = 0.2, origin = 'upper', interpolation='none')


    # Overlay contours for each cluster
    unique_clusters = np.unique(cluster_map)
    for cluster_id in unique_clusters:
        # Skip the background
        if cluster_id == -1:
            continue  
        mask = cluster_map == cluster_id  # Mask for the current cluster
        contours = find_contours(mask.astype(float), level=0.5)  # Find contours for the cluster
        for contour in contours:
            # Scale the contour coordinates to match the image axes
            new_contour = fix_contours(np.asarray(contour))
            plt.plot(new_contour[:, 1], new_contour[:, 0], color = 'red', lw = 5, label=f'Cluster {cluster_id}')  # Adjust axis direction
            
    cbar = fig.colorbar(im)
    cbar.set_label("Energy [MeV]", fontsize = 25)

    plt.title(title, fontsize = 22, loc = 'left', fontweight = 'bold')
    plt.axis('off')
    fig.savefig(f"Example_Cluster{num}.pdf", bbox_inches = 'tight')



def clustering_batch(images, eps, min_samples, threshold):

    ''' Function for clustering a batch of images. '''

    start_time = time()
    num_cluster = np.asarray([clustering_single_image(image, eps=eps, min_samples=min_samples, threshold=threshold)[0] for image in images])
    end_time = time()

    print(f"Processed {images.shape[0]} images in {end_time - start_time:.2f} seconds.")

    return num_cluster





def main():

    ''' Function for testing and plotting independent of the flow evaluation. '''

    num_samples = 10   # number of samples to perform clustering
    num_images = 5     # number of samples to plot 
    data_path = "/net/data_cms3a-1/kann/fast_calo_flow/data/shielding/"
    data = np.load(data_path + "test_img_photon.npy")[:num_samples]

    n_images, height, width = data.shape

    plot_examples = True

    if plot_examples:

        # Process a single example image for visualization
        for index, random_index in enumerate(np.random.randint(low = 0, high = num_samples, size = (num_images,))):
            example_image = data[random_index]
            num_clusters, cluster_map = clustering_single_image(
                example_image, eps=1.0, min_samples=1, threshold=400, visualize=True
            )

            # Plot the clusters with red borders if any were found
            if cluster_map is not None:
                plot_clusters(example_image, cluster_map, title=f"Found {num_clusters} Cluster(s)", num = index)
            else:
                print("No clusters found in the example image.")


    # Batch process all images for cluster counts
    start_time = time()
    cluster_counts = [clustering_single_image(data[i], eps=1.5, min_samples=1, threshold=400)[0] for i in range(n_images)]
    end_time = time()

    # Convert to NumPy array
    cluster_counts = np.array(cluster_counts)

    # Output
    print(f"Processed {n_images} images in {end_time - start_time:.2f} seconds.")
    print(f"Cluster counts per image: {cluster_counts[0:100]}")


if __name__ == "__main__":

    main()