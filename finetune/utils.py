import os

def count_file_num(folder_dir):
    num_png = 0
    num_npz = 0
    num_npy = 0
    
    for filename in os.listdir(folder_dir):
        if filename.endswith('.png'):
            num_png += 1
        elif filename.endswith('.npz'):
            num_npz += 1
        elif filename.endswith('.npy'):
            num_npy += 1

    assert num_png == num_npz or num_png == num_npy
    
    return num_png