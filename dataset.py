from argparse import ArgumentParser
import os
from utils import download_url
import tarfile
import shutil
from tqdm import tqdm
import numpy as np
from glob import glob

from torchvision import transforms
from PIL import Image


def join_path(arr):
    k = ''

    for i in arr:
        k = os.path.join(k, i)

    return k


def remove_empty_folders(base_path):
    for root, dirs, files in os.walk(base_path, topdown=False):
        for name in dirs:
            os.rmdir(join_path([root, name]))

    os.rmdir(base_path)


def structure_dataset(root_path, old_path, new_path, train_split):
    train_path = join_path([root_path, new_path, 'train'])
    val_path = join_path([root_path, new_path, 'val'])

    os.makedirs(train_path, exist_ok=True)
    os.makedirs(val_path, exist_ok=True)

    print('Processing data')
    print('Structuring dataset...')

    folders = os.listdir(join_path([root_path, old_path]))

    for folder in tqdm(folders):
        imgs = os.listdir(join_path([root_path, old_path, folder]))
        idxs = np.random.permutation(len(imgs))

        ims = np.floor(train_split * len(imgs)).astype(int)
        train_idx = idxs[:ims]
        val_idx = idxs[ims:]

        folder_name = folder.lower()

        for im in train_idx:
            os.makedirs(join_path([root_path, new_path, 'train', folder_name]), exist_ok=True)

            path_to_save = join_path([root_path, new_path, 'train', folder_name, imgs[im]])
            os.rename(join_path([root_path, old_path, folder, imgs[im]]), path_to_save)

        for im in val_idx:
            os.makedirs(join_path([root_path, new_path, 'val', folder_name]), exist_ok=True)

            path_to_save = join_path([root_path, new_path, 'val', folder_name, imgs[im]])
            os.rename(join_path([root_path, old_path, folder, imgs[im]]), path_to_save)

    remove_empty_folders(join_path([root_path, old_path]))


def calculate_stats(root_path, train_path):
    print('\n\nCalculating mean and standard deviation values...\n')
    train_files = np.array(glob(train_path))

    mstd = np.zeros((2, 3))

    to_tensor = transforms.ToTensor()

    for path in tqdm(train_files):
        img = Image.open(path)

        tensor = to_tensor(img)
        mstd[0] += np.mean(tensor.numpy(), axis=(1, 2))
        mstd[1] += np.std(tensor.numpy(), axis=(1, 2))

    mstd = mstd / len(train_files)
    save_path = join_path([root_path, 'stats.npy'])

    print('\n\nSaving to ' + save_path)
    np.save(save_path, mstd)


def load_stats(stats_path):
    '''
    Returns the mean and std of the dataset
    :param stats_path: Path where the mean and std is located
    :return: mean and std as numpy arrays
    '''
    stats = np.load(stats_path)

    return stats[0], stats[1]


def main(args):
    train_split = args.train
    root_path = args.path

    root_path = os.path.expanduser(root_path)
    filename = '101_ObjectCategories.tar.gz'

    download_url('http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz',
                 root_path,
                 filename,
                 "b224c7392d521a49829488ab0f1120d9")

    with tarfile.open(join_path([root_path, filename]), "r:gz") as tar:
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(tar, path=root_path)

    shutil.rmtree(join_path([root_path, '101_ObjectCategories', 'BACKGROUND_Google']))
    os.remove(join_path([root_path, filename]))

    structure_dataset(root_path, filename.split('.')[0], 'caltech_dataset', train_split)
    calculate_stats(root_path, join_path([root_path, 'caltech_dataset', 'train']) + '/*/*')


def parse_arguments():
    parser = ArgumentParser(description='Building the Caltech 101 dataset')

    parser.add_argument('--train',
                        '-t',
                        type=float,
                        default=0.8)

    parser.add_argument('--path',
                        '-p',
                        type=str,
                        default='data')

    return parser.parse_args()


if __name__ == '__main__':
    arguments = parse_arguments()
    main(arguments)
