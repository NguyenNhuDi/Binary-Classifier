import torch
from torch.utils.data import Dataset
import glob
import numpy as np
import pandas as pd
import multiprocessing as mp

csv_path = r'C:\Users\coanh\Desktop\Uni Work\First Model\labels.csv'
img_path = r'C:\Users\coanh\Desktop\Uni Work\First Model\Extracted Data'


class BinaryClassifierDataset(Dataset):
    def __init__(self, image_dir,
                 csv_dir,
                 epochs=1,
                 transform=None,
                 num_processes=1):
        # storing parameters
        self.image_dir = np.array(glob.glob(f'{image_dir}/*.jpg'))

        # Change the index to whatever the label folder has named it
        temp = pd.read_csv(csv_dir).to_dict()['cat']
        self.labels = np.array([temp[i] for i in temp])
        self.epochs = epochs
        self.transform = transform
        self.num_processes = num_processes

        # defining the joinable queues
        self.path_queue = mp.JoinableQueue()
        self.image_label_queue = mp.JoinableQueue()
        self.command_queue = mp.JoinableQueue()

        # defining the processes
        self.read_transform_processes = []

        for _ in range(num_processes):
            proc = mp.Process(target=BinaryClassifierDataset.__read_transform_image__,
                              args=(self.path_queue,
                                    self.command_queue,
                                    self.transform))
            self.read_transform_processes.append(proc)


    def __read_transform_image__(self, path_queue: mp.JoinableQueue,
                                 command_queue: mp.JoinableQueue):


if __name__ == '__main__':
    test = BinaryClassifierDataset(img_path, csv_path)
