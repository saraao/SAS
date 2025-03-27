import torch
import torch.nn
import numpy as np
import os
import os.path
import nibabel

import numpy as np
import cv2

class ACOMDataset(torch.utils.data.Dataset):
    def __init__(self, directory, test_flag=True, noise_rate=0):
        '''
        directory is expected to contain some folder structure.
        directory is named with imageid_layer.
        For example, for image id 1013 and difficulty_level 0 image,
        it has the directory:
        dataset/
            1013_0/
                - image.png
                - mask.png
                - newmask.png
            1013_1/
                - image.png
                - mask.png
                - newmask.png
        '''
        super().__init__()
        self.directory = os.path.expanduser(directory)        

        self.test_flag=test_flag

        if test_flag:
            self.seqtypes = ['image','newmask']
        else:
            self.seqtypes = ['image', 'mask','newmask']

        self.seqtypes_set = set(self.seqtypes)
        self.database = []

        #ACOM Dataset
        for root, dirs, files in os.walk(self.directory):
            # if there are no subdirs, we have data
            if not dirs:
                files.sort()
                datapoint = dict()
                # extract all files as channels
                for f in files:
                    seqtype = f.split('.')[0]
                    datapoint[seqtype] = os.path.join(root, f)
                self.database.append(datapoint)

        print("Current dataset directory:",self.directory)
        print("Current database len:",len(self.database))

        needsort_arr =[]
        for i in self.database:

            number_i = int(i['image'].split('/')[-2].split('_')[0])

            needsort_arr.append((number_i, i))

        result = sorted(needsort_arr, key=lambda t: t[0])
        new_result = []
        for i in result:
            new_result.append(i[1])

        self.database = new_result

        print("Current sorted database:",self.database[:10])

        #add noise data      
        from random import sample

        noise_directory = 'you/path/here/'
        noise_file_list = os.listdir(noise_directory)
        
        if noise_rate > 0.2:
            print("Error: Noise rate should be less than 0.2 and not smaller than 0!")
        else :
            print("noise rate is",noise_rate)
        
        noise_rate*=5  ## 20% noise file size is 100% noise_file_list

        full_len = len(noise_file_list)

        noise_len = int(full_len*noise_rate)

        sampled_noise_file_list = sample(noise_file_list,noise_len)
        
        print('Current sampled noise file list len:',len(sampled_noise_file_list))
        print('where 20% noise file size is',full_len)

        for dataset_name in sampled_noise_file_list:
            dataset_dir = os.path.join(noise_directory, dataset_name)
            for root, dirs, files in os.walk(dataset_dir):
                # if there are no subdirs, we have data
                if not dirs:
                    files.sort()
                    datapoint = dict()
                    # extract all files as channels
                    for f in files:
                        seqtype = f.split('.')[0]
                        datapoint[seqtype] = os.path.join(root, f)
                    self.database.append(datapoint)


    def __getitem__(self, x):
        out = []
        filedict = self.database[x]
            
        # Load an color image in grayscale
        img = cv2.imread(filedict['image'])
        img_ = img[:,:,::-1].transpose((2,0,1))

        if self.test_flag:
            # newmask is the previous layer mask
            layer_mask = cv2.imread(filedict['newmask'],cv2.IMREAD_GRAYSCALE)
            layer_mask_ = np.expand_dims(layer_mask, axis=0)

            layer_mask_[layer_mask_>0]=1

            new_image = np.append(img_, layer_mask_, axis=0)

            path=filedict['image']
            out = torch.tensor(new_image , dtype=torch.float32)
            image=out

            return (image, path)
        
        
        else:
            mask = cv2.imread(filedict['mask'],cv2.IMREAD_GRAYSCALE)
            mask_ = np.expand_dims(mask, axis=0)

            # newmask is the previous layer mask
            layer_mask = cv2.imread(filedict['newmask'],cv2.IMREAD_GRAYSCALE)
            layer_mask_ = np.expand_dims(layer_mask, axis=0)

            layer_mask_[layer_mask_>0]=1

            new_image = np.append(img_, layer_mask_, axis=0)
            
            new_image = np.append(new_image, mask_, axis=0)

            path=filedict['image']
            out = torch.tensor(new_image , dtype=torch.float32)

            image = out[:-1, ...]
            label = out[-1, ...][None, ...]

            # only class 0 and 1
            label=torch.where(label > 0, 1, 0).float()  #merge all tumor classes into one using binary mask

            return (image, label), ''


    def __len__(self):
        return len(self.database)