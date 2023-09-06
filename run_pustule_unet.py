'''
This script runs a directory of leaf images through a U-Net neural network to
create a segmentation mask of pustule vs. no pustule.

Typical usage:
python run_pustule_unet.py -mp path -pin path -pout filename -pltout path
    Options:
    -mp /home/cwhite/projects/leaf_tutorial/timecourse_models_v3/model_subset2_ni24_rs/
        Model Path (required). Path to the U-net model directory

    -pin /mnt/nvme2/cwhite/ml_leaf_tutorial/testing/full_tifs/
        Path to inputs (required). Path to directory that contains .tif images of leaves

    -pout /home/cwhite/projects/leaf_tutorial/run_leaves/run1.csv
        Path to output .csv file (required). Path to directory that will contain
        statistics calculated on each image. One file per directory.
        
    -pltout /home/cwhite/projects/leaf_tutorial/run_leaves/leaf_plots/
        Path to output debuggin figures (optional). If specified then figures with the
        original image, mask, and overlaid mask will be saved to this directory. This
        adds significantly to runtime. If no path specified, then plotting will be skipped, but
        stats will still be written to the .csv file.
        
    -t 0.2
        Threshold to use when turning the pustule probabilities into a binary mask. This is
        optional and only recommended to change if you want to be more or less strict with what
        is or is not determined to be a pusutle. If not specifed, then this script will use
        a previously 'tuned' thresholed determined by a hold-out dataset. It is note recommended
        to change this unless results with the tuned threshold are not ideal, or severely over/under
        predicting. By default this script looks for best_threshold.npy within the model folder.
'''


import glob
import numpy as np
import pandas as pd
from PIL import Image
import os
from scipy import ndimage
from matplotlib import pyplot as plt
import argparse

N = 8
os.environ["CUDA_VISIBLE_DEVICES"]="-1" 
os.environ["OMP_NUM_THREADS"] = f"{N}"
os.environ['TF_NUM_INTEROP_THREADS'] = f"{N}"
os.environ['TF_NUM_INTRAOP_THREADS'] = f"{N}"

import tensorflow as tf
tf.config.threading.set_inter_op_parallelism_threads(N)
tf.config.threading.set_intra_op_parallelism_threads(N)

import warnings
np.seterr(all='ignore')
pd.options.mode.chained_assignment = None
print('Tensorflow version ' + tf.__version__)

def extract_labels(labels, num_labels, orig_shape):
    num_labels = num_labels+1 # fixing something funky in the orig fn
    label_info = pd.DataFrame({})
    label_info['size'] = [np.sum(labels==label_num) for label_num in np.arange(num_labels)]
    label_info['id_num'] = [label_num for label_num in np.arange(num_labels)]
    label_info = label_info[label_info['size']!=np.max(label_info['size'])] # Remove largest label (background)
    return label_info

def reduce_resolution(img, factor=2):
    img = tf.image.resize(img, (img.shape[0]//factor, img.shape[1]//factor)).numpy().squeeze()
    return img
    
def increase_resolution(img, new_size):
    img = tf.image.resize(img, (new_size)).numpy().squeeze()
    return img


# Parse aguments
ap = argparse.ArgumentParser()
ap.add_argument('-mp', '--model_path', required=True, help='Path to the directory that contains the UNet Model')
ap.add_argument('-pin', '--input_image_path', required=True, help='Path to the direcotry that contains the .tif images of leaves you want to run')
ap.add_argument('-pout', '--output_csv_path', required=True, help='Path to .csv file that you want the output statistics to be stored.')
ap.add_argument('-pltout', '--output_figure_path', required=False, help='If --plotting True, then this is the path to the output dir for debug figures.')
ap.add_argument('-t', '--threshold', required=False, help='Value of the threshold used on the NN probabilites. If unspecified, pre-tuned threshold is used.')
args = vars(ap.parse_args())

MODEL_PATH = args['model_path']
PATH_TO_INPUT_IMAGES = args['input_image_path']
OUTPUT_CSV_FILE = args['output_csv_path']

# Parse debug figure path
PATH_TO_OUTPUT_PLOTS = args['output_figure_path']
if PATH_TO_OUTPUT_PLOTS is None: PLOTTING=False
else: PLOTTING= True
    
# Parse threshold
if args['threshold'] is None: probability_threshold = np.load(MODEL_PATH + 'best_threshold.npy')
else: probability_threshold = float(args['threshold'])
    
# Load model and grab image filenames
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
image_paths = np.sort(glob.glob(PATH_TO_INPUT_IMAGES + '*.tif'))
img_edge_size = 256 # Not a good idea to change this unless retraining the model

print()
print('Running with...')
print('Model: ', MODEL_PATH)
print('Input directory: ', PATH_TO_INPUT_IMAGES)
print('Output .csv file: ', OUTPUT_CSV_FILE)
print('Plotting: ', PLOTTING)
print('Plotting output: ', PATH_TO_OUTPUT_PLOTS)
print('Probability threshold: ', probability_threshold)
print()

########## Looping over all images in directory ##########
for image_number, image_filename in enumerate(image_paths):

    print(image_number, image_filename)
    image_filename = image_paths[image_number]

    # Read in image from .tif file 
    img = np.array(Image.open(image_filename), dtype='float32')

    # Get the dimensions of the image
    num_rows, num_cols = img.shape[0], img.shape[1]
    original_size = (num_rows, num_cols)

    # Before reducing resoltuion by a factor of two, this ensures that the have 
    # minimum image edges size of 512 by interpolating up slightly
    if ((num_rows<(img_edge_size*2)) | (num_cols<(img_edge_size*2))):
        min_dim = np.nanmin([num_rows,num_cols])
        factor = (img_edge_size*2)/min_dim
        new_size = (int(num_rows*factor)+1, int(num_cols*factor)+1)
        img = increase_resolution(img, new_size)
        num_rows, num_cols = img.shape[0], img.shape[1]

    img = reduce_resolution(img, factor=2)
    num_rows, num_cols = img.shape[0], img.shape[1]

    # This extracts tiles on a 512x512 grid. If there is some leftover
    # then there is a check to see how much leftovers there are. If its more than half the image size
    # then we tag on an extra row/col of tiles. 
    lower_row_bounds = np.arange(0,num_rows+1,img_edge_size)[0:-1]
    lower_row_bounds = np.append(lower_row_bounds, num_rows-img_edge_size)
    upper_row_bounds = lower_row_bounds + img_edge_size
    lower_col_bounds = np.arange(0,num_cols+1,img_edge_size)[0:-1]
    lower_col_bounds = np.append(lower_col_bounds, num_cols-img_edge_size)
    upper_col_bounds = lower_col_bounds + img_edge_size

    # Set up the the arrays to accept the interpolated tiles
    total_num = (len(lower_row_bounds)*len(lower_col_bounds))
    tiled_images = np.zeros([total_num, img_edge_size, img_edge_size, 3],dtype='float32')

    #Loop over the image and crop the tiles
    k=0
    for i in range(lower_row_bounds.size):
        for j in range(lower_col_bounds.size):
            lower_row_bound = lower_row_bounds[i]
            upper_row_bound = upper_row_bounds[i]
            lower_col_bound = lower_col_bounds[j]
            upper_col_bound = upper_col_bounds[j]
            crop_tile = img[lower_row_bound:upper_row_bound,lower_col_bound:upper_col_bound]
            tiled_images[k] = np.array(crop_tile/255.0,dtype='float32')
            k+=1

    # Make predicitions with the neural network
    # Add test time augmentation here if wanted later
    predictions = model.predict(tiled_images, batch_size=4, verbose=0).squeeze()
    nn_predicted_probs = np.zeros([img.shape[0], img.shape[1]])

    #Loop over the predcitions and insert into img-shaped array
    k=0
    for i in range(lower_row_bounds.size):
        for j in range(lower_col_bounds.size):
            lower_row_bound = lower_row_bounds[i]
            upper_row_bound = upper_row_bounds[i]
            lower_col_bound = lower_col_bounds[j]
            upper_col_bound = upper_col_bounds[j]
            nn_predicted_probs[lower_row_bound:upper_row_bound,lower_col_bound:upper_col_bound] = predictions[k]
            k+=1

    # Increase the resolution back to original dimensions
    nn_predicted_probs = increase_resolution(nn_predicted_probs[:,:,None], original_size)
    img = increase_resolution(img, original_size)
    nn_predicted = nn_predicted_probs>probability_threshold

    ########## Create comparison dictionary for this image ##########

    # Extract the NN label information so we can eventually combine it with the truth
    labels,num_labels = ndimage.label(nn_predicted_probs>probability_threshold)
    label_info = extract_labels(labels, num_labels, original_size)

    stats_dict = {'image_name': os.path.basename(image_filename)[0:-4],
     'num_image_px': img[:,:,0].size,
     'image_row_dim': original_size[0],
     'image_col_dim': original_size[1],
     'num_nn_pusutle_px': np.sum(nn_predicted==1),
     'num_nn_pustules': len(np.unique(labels))-1,
     'leaf_area_approx_px': np.sum(img[:,:,2]>160.0),
                          }

    stats_dict['percent_coverage'] = (stats_dict['num_nn_pusutle_px'] / stats_dict['leaf_area_approx_px'])*100

    if len(label_info)!=0:
        stats_dict['min_pustule_size'] = np.nanmin(label_info['size'])
        stats_dict['max_pustule_size'] = np.nanmax(label_info['size'])
        stats_dict['mean_pustule_size'] = np.nanmean(label_info['size'])
        stats_dict['median_pustule_size'] = np.nanmedian(label_info['size'])
    else:
        stats_dict['min_pustule_size'] = np.nan
        stats_dict['max_pustule_size'] = np.nan
        stats_dict['mean_pustule_size'] = np.nan
        stats_dict['median_pustule_size'] = np.nan

    print('Numnber of Pustules: ', stats_dict['num_nn_pustules'])
    print('Percent Coverage: ', stats_dict['percent_coverage'], '%')
    print('Mean Size: ', stats_dict['mean_pustule_size'], 'pixels')
    print('Median Size: ', stats_dict['median_pustule_size'], 'pixels')

    if PLOTTING==True:
        f, (a0, a1, a2) = plt.subplots(3, 1, gridspec_kw={'height_ratios': [1, 1.24, 1]}, figsize=(48,16))
        
        plt.subplot(311)
        plt.imshow(img/255.0, interpolation='nearest')
        plt.title(image_filename, fontsize=18)
        plt.yticks(np.arange(0,img.shape[0],256))
        plt.xticks(np.arange(0,img.shape[1],256))
        plt.grid(alpha=0.6)
        
        plt.subplot(312)
        plt.imshow(nn_predicted_probs, interpolation='nearest', vmin=0,vmax=1,cmap='magma')        
        plt.yticks(np.arange(0,img.shape[0],256))
        plt.xticks(np.arange(0,img.shape[1],256))
        plt.grid(alpha=0.3)
        plt.colorbar(label='Pustule Class Probability', orientation='horizontal', pad=0.14, fraction=0.05)
        
        plt.subplot(313)
        plt.imshow(img/255.0, interpolation='nearest')
        plt.imshow(nn_predicted,alpha=0.5, interpolation='nearest', vmin=0,vmax=1, cmap='magma')
        plt.yticks(np.arange(0,img.shape[0],256))
        plt.xticks(np.arange(0,img.shape[1],256))
        plt.grid(alpha=0.3)
        plt.savefig(PATH_TO_OUTPUT_PLOTS + f'/{os.path.basename(image_filename)}' + '.png', dpi=150, bbox_inches='tight')
        plt.clf()
        plt.close()

    stats_df = pd.DataFrame(stats_dict, index=[image_number])
    if image_number==0: 
        running_pustule_df = stats_df.copy()
    else: 
        running_pustule_df = pd.concat([running_pustule_df, stats_df])
        running_pustule_df.to_csv(OUTPUT_CSV_FILE)

    print('-'*30)
    print()