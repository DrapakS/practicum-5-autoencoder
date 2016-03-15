import pickle
from display_layer import *


def normalize_data(images):
    """
    :param images: numpy array, image to normalize
    :return: normalized image. For each channel we transform data to [mean - 3*std, mean + 3*std].
                               Same dim as images
    """
    d = int(np.sqrt(images.shape[1]/3))
    N = images.shape[0]
    shaped_images = np.reshape(images, (N, d, d, 3)).copy()
    r_std = np.std(shaped_images[:, :, :, 0])
    g_std = np.std(shaped_images[:, :, :, 1])
    b_std = np.std(shaped_images[:, :, :, 2])

    r_mean = np.mean(shaped_images[:, :, :, 0])
    g_mean = np.mean(shaped_images[:, :, :, 1])
    b_mean = np.mean(shaped_images[:, :, :, 2])

    shaped_images[:, :, :, 0] = np.clip(shaped_images[:, :, :, 0], r_mean - 3 * r_std, r_mean + 3 * r_std)
    shaped_images[:, :, :, 1] = np.clip(shaped_images[:, :, :, 1], g_mean - 3 * g_std, g_mean + 3 * g_std)
    shaped_images[:, :, :, 2] = np.clip(shaped_images[:, :, :, 2], b_mean - 3 * b_std, b_mean + 3 * b_std)

    return np.reshape(shaped_images, (N, d * d * 3))


def download_data(list_of_files):
    """
    support function to load data from pickle files from list_of_files
    :param list_of_files: list of files to load
    :return: numpy array of images, labels of images
    """
    result = np.array([])
    labels = np.array([])
    for file_name in list_of_files:
        file = open(file_name, 'r')
        data = pickle.load(file)
        file.close()
        if type(data) == type(np.array([])):
            data = np.array(data, dtype='uint8')
        else:
            if 'y' in data.keys():
                labels = data['y']
            data = np.array(data['X'], dtype='uint8')

        if result.shape[0] == 0:
            result = data
        else:
            result = np.hstack((result, data))
    return result, labels


def sample_patches_raw(images, num_patches=10000, patch_size=8):
    """
    :param images: numpy array of images
    :param num_patches:  number of patches to sample
    :param patch_size:  size of each patches
    :return: numpy array of patches (num_patches, 3 * patches_size)
    """
    lin_pathces_size = patch_size * patch_size * 3
    patches = np.empty((num_patches, lin_pathces_size), dtype='uint8')
    M = int(np.sqrt(images.shape[1]/3))
    N = M

    im_num = np.random.randint(0, images.shape[0], num_patches)
    start_lines = np.random.randint(0, N - patch_size - 1, num_patches)
    start_cols = np.random.randint(0, M - patch_size - 1, num_patches)
    for i in range(num_patches):
        im = np.reshape(images[im_num[i]], (M, N, 3))
        patches[i] = np.reshape(im[start_lines[i]:start_lines[i] + patch_size, start_cols[i]:start_cols[i] + patch_size, :], (lin_pathces_size))
    return patches


def sample_patches(images, num_patches=10000, patch_size=8):
    """
    :param images: numpy array of images
    :param num_patches: int, number of patches to sample
    :param patch_size: int, linear size of each patches (8 for 8x8 patches)
    :return: numpy array of patches (num_patches, 3 * patches_size),
                    normalized with normalize data function
    """
    patches = sample_patches_raw(images, num_patches, patch_size)
    patches = normalize_data(patches)
    return patches


def no_system_overload_sample_patches(filelist, num_patches=10000, patch_size=8, normalize=False):
    """
    support function to save load patches from several separated files
    :param filelist: list of files to load
    :param num_patches: int, number of patches to sample
    :param patch_size: int, linear size of each patches (8 for 8x8 patches)
    :param normalize: normalize data if True
    :return: numpy array of patches (num_patches, 3 * patches_size)
    """
    n_files = len(filelist)
    file_numbers = np.random.randint(0, n_files, num_patches)
    patches = np.empty((num_patches, patch_size * patch_size * 3))
    sample_f = sample_patches_raw
    if normalize:
        sample_f = sample_patches
    for i in range(len(filelist)):
        mask = np.where(file_numbers == i)[0]
        images, _ = download_data([filelist[i]])
        patches[mask] = sample_f(images, mask.shape[0], patch_size)
    return patches
