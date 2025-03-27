import argparse
import os
import nibabel as nib
import sys
import random
sys.path.append(".")
import numpy as np
import time
import torch as th
import torch.distributed as dist
from tqdm import tqdm
import torchmetrics
import shutil
import torch
from PIL import Image
from matplotlib import cm
import cv2

from guided_diffusion import dist_util, logger

from guided_diffusion.acomloader import ACOMDataset

from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
seed=10
th.manual_seed(seed)
th.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)


def visualize(img):
    _min = img.min()
    _max = img.max()
    normalized_img = (img - _min)/ (_max - _min)
    return normalized_img

def dice_score(pred, targs):
    pred = (pred>0).float()
    return 2. * (pred*targs).sum() / (pred+targs).sum()


def create_argparser():
    defaults = dict(
        data_dir="",
        clip_denoised=True,
        num_samples=1,
        batch_size=1,
        use_ddim=False,
        model_path="",
        predict_save_dir = "",
        num_ensemble=3,      #number of samples in the ensemble
        # number of the maximum generated layers, if the number of layers exceeds this number, 
        # the generation of the current image will be terminated.
        # Use the maximum number of layers from the training set
        max_layer = 5        
    )
    defaults.update(model_and_diffusion_defaults())

    print('model_and_diffusion_defaults(),', model_and_diffusion_defaults())

    parser = argparse.ArgumentParser()
    print('paraser,',parser)
    add_dict_to_argparser(parser, defaults)
    return parser


def get_sample_data(data_path):
        out = []
        # Load an color image in grayscale
        img = cv2.imread(data_path['image'])
        img_ = img[:,:,::-1].transpose((2,0,1))
        # newmask is the previous layer mask
        layer_mask = cv2.imread(data_path['newmask'],cv2.IMREAD_GRAYSCALE)
        layer_mask_ = np.expand_dims(layer_mask, axis=0)
        layer_mask_[layer_mask_>0]=1
        new_image = np.append(img_, layer_mask_, axis=0)
        path=data_path['image']
        out = torch.tensor(new_image , dtype=torch.float32)
        image=out

        return (image, path)



if __name__ == "__main__":

    args = create_argparser().parse_args()
    
    dist_util.setup_dist()

    print(args)

    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    average_precision = torchmetrics.AveragePrecision(task="binary")
    folder_name = args.data_dir
    file_names = os.listdir(folder_name)
    predict_path = args.predict_save_dir
    handle_predict_path = predict_path+'_combine'

    print('predict_path,',predict_path)
    print('handle_predict_path,',handle_predict_path)

    file_names = []
    for i in os.listdir(folder_name):
        file_names.append(i.split('_')[0])
    file_names = list(set(file_names))
    file_names.sort()

    layer='0'

    ds = ACOMDataset(args.data_dir, test_flag=True, noise_rate=0)
    datal = th.utils.data.DataLoader(
        ds,
        batch_size=1,
        shuffle=False)
    data = iter(datal)

    print("Max Layer:", args.max_layer)

    all_images = []
    print('len(ds),', len(ds))

    ## gen data
    while True:
        try:
            b, path = next(data)  #should return an image from the dataloader "data"            
            c = th.randn_like(b[:, :1, ...])
            img = th.cat((b, c), dim=1)     #add a noise channel$

        except:
            print('end or error')
            break

        slice_ID=path[0].split("/", -1)[-2]
        logger.log("sampling...")
        image_name = slice_ID.split('_')[0]

        current_layer = 0

        while True:

            layer = str(current_layer)

            if current_layer != 0:
                data_path={}
                data_path['mask']=handle_predict_path+'_'+str(int(current_layer)-1)+'/'+image_name+'/mask.png'
                data_path['newmask']=handle_predict_path+'_'+str(int(current_layer)-1)+'/'+image_name+'/newmask.png'
                data_path['image']=handle_predict_path+'_'+str(int(current_layer)-1)+'/'+image_name+'/image.png'
                print(data_path)

                b, path = get_sample_data(data_path)
                b = torch.unsqueeze(b, dim=0)
                c = th.randn_like(b[:, :1, ...])
                img = th.cat((b, c), dim=1)     #add a noise channel$
            
            start = th.cuda.Event(enable_timing=True)
            end = th.cuda.Event(enable_timing=True)

            for i in range(args.num_ensemble):  #this is for the generation of an ensemble of number=num_ensemble masks.
                print('start ...... ')
                model_kwargs = {}
                start.record()
                sample_fn = (
                    diffusion.p_sample_loop_known if not args.use_ddim else diffusion.ddim_sample_loop_known
                )

                sample, x_noisy, org = sample_fn(
                    model,
                    (args.batch_size, 3, args.image_size, args.image_size), img,
                    clip_denoised=args.clip_denoised,
                    model_kwargs=model_kwargs,
                )

                end.record()
                th.cuda.synchronize()
                print('time for 1 sample', start.elapsed_time(end))  #time measurement for the generation of 1 sample
                print('saved as'+ str(slice_ID))
                s = th.tensor(sample)

                os.makedirs(predict_path + '_' + layer +'/', exist_ok = True)


                if layer =='0':
                    th.save(s, predict_path + '_' + layer +'/' +str(slice_ID)+'_output'+str(i)) #save the generated mask
                    print('output predict save to: ', predict_path + '_' + layer +'/' +str(slice_ID)+'_output'+str(i))
                else:
                    th.save(s, predict_path + '_' + layer +'/' +image_name+'_'+layer+'_output'+str(i)) #save the generated mask
                    print('output predict save to: ', predict_path + '_' + layer +'/' +image_name+'_'+layer+'_output'+str(i))

            break_flag = False
            mask_arr=[]
            Ensemble=[]
            # Load the number=num_ensemble predictions of each image
            # by running this block of code number=num_ensemble times
            for i in range(args.num_ensemble):
                # Creates a path string to the output file
                out_path = predict_path + '_' + layer+'/'+image_name+'_'+layer+'_output'+str(i)
                print('output_path range 3: ', out_path)
                
                # Checks if the output file exists or breaks the loop
                if not os.path.isfile(out_path):
                    break_flag=True
                    break
                
                # If the file exists, loads the tensor data from the file into the out_i variable
                # applies a threshold to the tensor data, and stores the processed data in the Ensemble list
                out_i = torch.load(out_path, map_location=torch.device('cpu'))


                # Rescale the prediction to [0,1]
                # If the prediction is less than 0, set it to 0
                # Otherwise, keep the original value
                tensor = out_i
                threshold_mask = out_i < 0
                tensor[threshold_mask] = 0
                threshold_mask = out_i < 0.5
                tensor[threshold_mask] = 0
                threshold_mask = out_i >= 0.5
                tensor[threshold_mask] = 1

                mask = tensor.squeeze().cpu().numpy()
                if mask.mean() < 0.0025:
                    continue

                Ensemble.append(mask)
            
            #if mask mean ==0 
            if Ensemble ==[]:
                break

            # If the output file does not exist, 
            # skips the rest of the current loop iteration and starts the next one.
            if break_flag:
                continue
            
            # Compute the variance map of the ensemble prediction
            var = torch.Tensor(np.array(Ensemble).var(axis=0)) 

            # Compute the mean map of the ensemble prediction
            mean = torch.Tensor(np.array(Ensemble).mean(axis=0))

            # Rescale the mean map to [0,1]
            # If the mean map is less than 0, set it to 0
            # Otherwise, keep the original value
            mean_prediction = mean

            threshold_mask = mean_prediction < 0
            mean_prediction[threshold_mask] = 0
            mean_prediction_mask = mean_prediction.squeeze()
            mean_prediction_maskgray_image = np.uint8(mean_prediction_mask * 255)  # Convert to 8-bit pixel values
            mean_prediction_maskgray_image_mean_map = cv2.cvtColor(mean_prediction_maskgray_image, cv2.COLOR_GRAY2RGB)

            mask_arr = []
            show_gray_output_image_arr = []

            # Rescale the prediction to [0,1]
            # If the prediction is less than 0.5, set it to 0
            # Otherwise, set it to 1
            for number_i in range(len(Ensemble)):
                Ensemble_sub = torch.Tensor(Ensemble[number_i].copy())
                threshold_mask = Ensemble_sub < 0.5
                Ensemble_sub[threshold_mask] = 0
                threshold_mask = Ensemble_sub >= 0.5
                Ensemble_sub[threshold_mask] = 1
                mask = Ensemble_sub.squeeze()
                mask_arr.append(mask)
                gray_output_image = np.uint8(mask * 255)  # Convert to 8-bit pixel values
                show_gray_output_image_arr.append(gray_output_image)

            # Compute the absolute difference between the mean map and each prediction
            abs_mask_arr_score = []
            # find the index of the mask with the smallest absolute difference
            for index, sub_mask_arr in enumerate(mask_arr):
                abs_mask_arr_score.append(torch.abs(sub_mask_arr-mean_prediction_mask).sum())

            get_min_mask_index = np.where(abs_mask_arr_score==np.min(abs_mask_arr_score))[0][0]

            # this layer mask is the mask with the smallest absolute difference
            layer_mask =  show_gray_output_image_arr[get_min_mask_index]

            # if leayer_mask is all 0, skip this image and continue to the next image
            if layer_mask.mean()==0:
                break
            
            # current layer predict
            predict_mask = layer_mask.copy()
            # newmask is used to input to model predic. as all the previous predict
            # predicmask is currently layer predic 
            # if layer !='0', add the last layer mask to the current layer mask
            if layer != '0':
                lastlayer_maskpath  = handle_predict_path+'_'+str(int(layer)-1)+'/'+image_name+'/newmask.png'
                last_layer_mask =  cv2.imread(lastlayer_maskpath, cv2.IMREAD_GRAYSCALE)
                layer_mask = layer_mask + last_layer_mask

                lastpridic_maskpath = handle_predict_path+'_'+str(int(layer)-1)+'/'+image_name+'/predicmask.png'
                lastpridic_mask =  cv2.imread(lastpridic_maskpath, cv2.IMREAD_GRAYSCALE)

                is_samelayer = np.logical_and(predict_mask,lastpridic_mask)
                if True not in is_samelayer:
                    newlastpridic_mask = lastpridic_mask + predict_mask
                    ### update last layer predicmask
                    cv2.imwrite(lastpridic_maskpath, newlastpridic_mask)
                    ### update last layer new mask
                    cv2.imwrite(lastlayer_maskpath, layer_mask)

                    continue


            out_path = handle_predict_path+'_'+layer+'/'+image_name
            os.makedirs(out_path, exist_ok = True)

            new_img_file = out_path+'/predicmask.png'
            cv2.imwrite(new_img_file, predict_mask)

            # writes the processed image to a file
            # and copies the original image to the same directory
            new_img_file = out_path+'/newmask.png'
            cv2.imwrite(new_img_file, layer_mask)

            src = folder_name+image_name+'_0/image.png'
            dst = out_path+'/image.png'
            shutil.copyfile(src, dst)

            current_layer+=1

            if current_layer> args.max_layer:
                break
