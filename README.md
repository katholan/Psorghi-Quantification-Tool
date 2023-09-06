## Psorghi Quantification Tool

Pustule area quantification on single maize leaves utilizing a U-Net convolutional neural network model. This tool is intended for use on leaves imaged on a flatbed scanner with a blue background. It runs a directory of leaf images through a U-Net neural network to create a segmentation mask of pustule vs. no pustule.

### Data Prep

Acquire images of your maize leaves.

Separate your leaves so that there is a single leaf on each image. You can use `separating_leaves.ipynb` (a Jupyter Python3 notebook) for this purpose. There are comments within this file on how to set it up and run for your images. I run Jupyter through Anaconda on my local machine.

Place your cropped photos into their own folder.

Create a folder for your output `.csv` files. The `.csv` files will contain your U-Net results for each image.

Create a folder for your output `.tif` files. The `.tif` files will be the original image, the model's probabilities (that each pixel is a pustule), the model's predictions, and the model's predictions overlaid on the original image.

### Installation

Clone this GitHub repository locally on your machine.

```git clone https://github.com/katholan/Psorghi-Quantification-Tool.git```

Create a conda environment with the provided `.yml` file.

```conda env create -f environment.yml```

Activate your new environment

```conda activate env``` or ```source activate env```

### Running the code

Typical usage:

```python run_pustule_unet.py -mp path/to/model/model_utc4-510 -pin path/to/inputs -pout path/output/csv/filename -pltout path/output/figures```

Options:

    Required
    -mp #Model Path (required). Path to the U-net model directory

    -pin #Path to inputs (required). Path to directory that contains .tif images of leaves

    -pout #Path to output .csv file (required). Path to directory that will contain
        statistics calculated on each image. Output will be one row per image and one file per directory.


    Optional
    
    -pltout #Path to output debugging figures (optional). If specified then figures with the
        original image, mask, and overlaid mask will be saved to this directory. This
        adds significantly to runtime. If no path specified, then plotting will be skipped, but
        stats will still be written to the .csv file.
        
    -t #Value between 0 and 1. eg 0.2.
        Threshold to use when turning the pustule probabilities into a binary mask. This is
        optional and only recommended to change if you want to be more or less strict with what
        is or is not determined to be a pusutle. If not specifed, then this script will use
        a previously 'tuned' thresholed determined by a hold-out dataset. It is not recommended
        to change this unless results with the tuned threshold are not ideal, or severely over/under
        predicting. By default this script looks for best_threshold.npy within the model folder.


Example statistics output is at `run1.csv` and example debugging figures are available at `leaf_plots/`. The original .tif images for these leaves are available at `raw_images`.
