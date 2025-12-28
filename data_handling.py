import torch
from torchvision.io import read_image
from torchvision.transforms import Resize
import os


class DataConverter:

    def __init__(self,verbose=True):

        self.verbose = verbose
        self.image_prefixes = ['dataset//Non-COVID-19//Non-COVID-19_',
                          'dataset//COVID-19//COVID-19_']
        
        self.inputs = []
        self.labels = []
        self.num_imgs = [0,0]

        if not self.verbose:
            print('Loading Data')

        for i , pref in enumerate(self.image_prefixes):

            img_number = 0
            image_exists = True

            while image_exists:

                img_number += 1
                number_prefix = ''

                if i == 1:
                    if img_number < 1000:
                        number_prefix += '0'
                
                if img_number < 100:
                    number_prefix += '0'
                if img_number < 10:
                    number_prefix += '0'

                img_file = pref + number_prefix + str(img_number) + '.png'
                image_exists = os.path.exists(img_file)

                if image_exists:
                    self.num_imgs[i] += 1
                    pic = read_image(img_file)
                    self.inputs.append(pic)
                    self.labels.append(i)

            if self.verbose:
                if i == 0:
                    print(f'Non Covid Images: {self.num_imgs[i]}')
                else:
                    print(f'Covid Images: {self.num_imgs[i]}')

        self.N = sum(self.num_imgs)

        if self.verbose:
            print(f'Total number of images: {self.N}')
            print(f'Percentages of Covid & Non Covid photos: {round((self.num_imgs[1] / self.N)*100,3)}% , {round((self.num_imgs[0] / self.N)*100,3)}%\n')

        self.unnorm_inputs = torch.stack(self.inputs).to(dtype=torch.float32).detach()
        self.norm_inputs = self.unnorm_inputs / 255.
        self.labels = torch.tensor(self.labels).long().detach()

        if self.verbose:
            print(f'Input data dimensions: {self.unnorm_inputs.size()}')
            print(f'Labels dimensions: {self.labels.size()}\n')

    def train_test_splitting(self,split_train):

        print(f'Shuffling data and splitting at {split_train}:{(1 - split_train):.2f} ratio')
        
        split_train_index = int(self.N * split_train)
        perm = torch.randperm(self.N)

        shuffled_inputs = self.norm_inputs[perm]
        shuffled_labels = self.labels[perm]

        train_inputs, test_inputs = shuffled_inputs[:split_train_index], shuffled_inputs[split_train_index:]
        train_labels, test_labels = shuffled_labels[:split_train_index], shuffled_labels[split_train_index:]

        train_size = train_inputs.size(0)
        test_size = test_inputs.size(0)

        if self.verbose:
            print(f'Training size: {train_size}')
            print(f'Testing size: {test_size}\n')

        frac_non_covids_train , frac_covids_train = (train_labels==0).sum().item() / train_size , (train_labels==1).sum().item() / train_size
        frac_non_covids_test , frac_covids_test = (test_labels==0).sum().item() / test_size , (test_labels==1).sum().item() / test_size

        if self.verbose:
            print(f'Percentage of Non Covids in Training data: {round(frac_non_covids_train*100,3)}%')
            print(f'Percentage of Covids in Training data: {round(frac_covids_train*100,3)}%\n')
            print(f'Percentage of Non Covids in Testing data: {round(frac_non_covids_test*100,3)}%')
            print(f'Percentage of Covids in Testing data: {round(frac_covids_test*100,3)}%\n')

        return train_inputs, train_labels, test_inputs, test_labels