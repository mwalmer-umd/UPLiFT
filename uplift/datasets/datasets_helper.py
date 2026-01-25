"""
Helper for loading datasets with default paths.

Modify this script to specify the paths to your datasets, or to add new datasets.
"""

def get_dataroot(dataset):
    if dataset == 'imagenet':
        dataroot = '/your/path/to/dataset/'
    elif dataset == 'unsplash_lite':
        dataroot = '/your/path/to/dataset/'
    else:
        print('ERROR: dataset %s not recognized'%dataset)
        print('to load this dataset, add its root path to datasets/datasets_helper.py')
        exit(-1)
    return dataroot