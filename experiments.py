from autoencoder import *
from gradient import *


def get_subimage_feature(images, step, patch_size, theta, visible_size, hidden_size, layer_number):
    """
    support function for experiments. Compute feature for classification. Divide image on subimages
    and apply transform for each of them. Then concatenate it to feature vector.
    """
    d = int(np.sqrt(images.shape[1]/3))
    N = images.shape[0]
    new_images = np.array([])
    for i in range(N):
        image = images[i].reshape((d, d, 3))
        new_image = np.array([])
        for j in range((d - patch_size)/step):
            for k in range((d - patch_size)/step):
                patch = image[j * step:j * step + patch_size, k * step:k * step + patch_size].flatten()
                if new_image.shape[0] == 0:
                    new_image = patch
                else:
                    new_image = np.vstack((new_image, patch))
        #print new_image.shape
        new_image = autoencoder_transform(theta, visible_size, hidden_size, layer_number, new_image).flatten()
        if new_images.shape[0] == 0:
            new_images = new_image
        else:
            new_images = np.vstack((new_images, new_image))
    return new_images
