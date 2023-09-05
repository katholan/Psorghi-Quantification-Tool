## Psorghi Quantification Tool

Pustule area quantification on single maize leaves utilizing a U-Net convolutional neural network model. This tool is intended for use on leaves imaged on a flatbed scanner with a blue background.

### Data Prep

Acquire images of your maize leaves.

Separate your leaves so that there is a single leaf on each image. You can use `separating_leaves.ipynb` (a Jupyter Python3 notebook) for this purpose. There are comments within this file on how to set it up and run for your images. I run Jupyter through Anaconda on my local machine.

Place your cropped photos into their own folder.

Create a folder for your output `.csv` files. The `.csv` files will contain your U-Net results for each image.

Create a folder for your output `.tif` files. The `.tif` files will be the original image, the model's probabilities (that each pixel is a pustule), the model's predictions, and the model's predictions overlaid on the original image.

### Installation

Clone this GitHub repository locally on your machine.

```git clone REPOSITORY```

Create a conda environment with the provided `.yml` file.

```conda env create -f environment.yml```

Activate your new environment

```conda activate env``` or ```source activate env```

### Running the code

```python3 FILE -input_folder -output_csvs -output_tifs -threshold```

```
input_folder #folder containing your leaf images
output_csvs #folder where your results files will be saved
output_tifs #folder where your results images will be saved. If not specified, these will not be saved.
threshold #option to change the threshold for what is considered a pustule. If left blank, it will use the best threshold as determined by the validation set during the training of this U-Net model (Recommended). Default is XxX. Use a lower value to be stricter, or a higher value to be more lenient.
```

Run time will depend on machine specs, but should run fairly quickly.
