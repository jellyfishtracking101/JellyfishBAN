# BrainAlignNet
BrainAlignNet is a deep neural network that registers neurons in the deforming head of freely-moving C. elegans. This repository contains the source code for data preprocessing, as well as network training and testing.

## citation
To cite this work, please refer to our preprint:

**Deep Neural Networks to Register and Annotate the Cells of the C. elegans Nervous System**

Adam A. Atanas, Alicia Kun-Yang Lu, Jungsoo Kim, Saba Baskoylu, Di Kang, Talya S. Kramer, Eric Bueno, Flossie K. Wan, Steven W. Flavell

bioRxiv 2024.07.18.601886; doi: https://doi.org/10.1101/2024.07.18.601886

## table of contents
- [installation](#installation)
- [data preparation](#data-preparation)
    - [cropping & Euler registration](#cropping--euler-registration)
    - [create centroids](#create-centroids)
- [usage](#usage)

## installation
BrainAlignNet runs on two other packages: `DeepReg` and `euler_gpu`, which need to be installed separately - make sure to use the Flavell Lab specific installations.

### DeepReg

`DeepReg` is a deep learning toolkit for image registration. BrainAlignNet uses a [custom version of `DeepReg`](https://github.com/flavell-lab/DeepReg) with a novel network objective.

Clone or download our custom DeepReg; then run `pip install .` at its root directory to install the package.

### euler_gpu

`euler_gpu` is a GPU-accelerated implementation of Euler registration using pytorch.

Clone or download [`euler_gpu`](https://github.com/flavell-lab/euler_gpu) and run `pip install .` at the root directory.

## data preparation

*For a demonstration of our data preprocessing pipeline, check out our [demo notebook](https://github.com/flavell-lab/BrainAlignNet/blob/main/demo_notebook/demo_pipeline.ipynb).*


The inputs to BrainAlignNet are images with their centroid labels. Each registration problem in the training and validation set is composed of six items:
* `fixed_image` & `moving_image`
* `fixed_roi` & `moving_roi`
* `fixed_label` & `moving_label`

These datasets for all registration problems are written in `.h5` files. Each `.h5` file contains multiple keys. Each key, formatted as `<t_moving>to<t_fixed>`, represents a registration problem.

During training, BrainAlignNet is tasked with optimally registering the `t_moving` frame to the `t_fixed` frame, where `t_moving` and `t_fixed` are two different timepoints from a single calcium imaging recording.

The ROI images, `fixed_roi` and `moving_roi`, display each neuron on the RFP images with a unique color. Each label is a list of centriods of these neuronal R0Is.

To prepare the data for training and validation, the run the `sandwich_making.ipynb` script.

## TODO: Docs on Sandwich



## usage

*The original demonstration of training and applying BrainAlignNet on unseen data is available [here](https://github.com/flavell-lab/BrainAlignNet/blob/main/demo_notebook/demo_network.ipynb)*.

### RFa Inference Pipeline

This is everything I did to align the most recent video I ran. This process assumes that you have already trained a dataset-specific or generalized model, and just runs through the inference pipeline. As of 5/20/25,this entire pipline has been ran and validated exclusively using the following steps.

1. Chunk a video into sections for easier loading and processing. I used 2000 frames.  
   1. I have a ImageJ macro for this if it would be helpful  
2. Open **eulerGPU-rigalign RFa.ipynb**[^1]  
   1. Update the `red_channel` and `BATCH_SIZE` variables in the first cell to match your setup. The “red\_channel” doesn’t have to be literally red, but is the channel that is constant that we want to use for alignment. Note it is zero-indexed, so the first channel is channel 0\.  
   2. Update the `tiff_folder` path in the second cell to match the location of your chunked video. Double check the files it prints out are correct and in the correct order.  
   3. Run All Cells \- (this can take several hours)  
   4. Evaluate the output. If it’s poorly aligned, try to figure out which frame caused it to get out of sync. If there’s a lot of movement on that frame, increase the search range values or uncommenting the expanded search snippet in the main cell  
3. Open **alignVideoSecondChannelOneTIF-RFa.ipynb**  
   1. Update the batch size to account for what you can fit on your GPU (although inference is less memory-hungry, it might be best to just use what you used in training)  
   2. Update `tiff_folder` in the second cell to match where your rigid-aligned video is. This should be the same as the folder in part 2b. Using the third cell, double check the files are just the files created from step 2  
   3. Update `checkpoint_path` to match your best checkpoint from training. The path should end with “ckpt-000” where 000 is whatever epoch was best. Don’t include an extension  
   4. Set `max_frames` to the size of your chunk (e.g. 2000\)  
   5. Check that `side_len` matches your crop size. If you have non-square images, then you’ll have to modify my code. You might also want to check beforehand that none of the zeros added during rigid-alignment are included in the crop, since that will mess up the scaling. If that’s the case and you can’t re-crop, try subtracting the median and then taking the max of that array and 0\.  
   6. Make sure `red_chan` is correct as mentioned previously  
   7. If your image is double padded in the default way, set `padding` to 

      `[[0,0],[0,0],[0,0],[1,1]]`

      Else, if it’s single padded in the default way, set it to 

      `[[0,0],[0,0],[0,0],[1,0]]`

      And set `z_depth` to 2 or 3 depending on padding

   8. `log_dir` needs to be a valid path, but don’t worry about it too much  
   9. Run all cells (depending on your system’s read-write speed and load, I’ve had this take anywhere from 20min to 3+ hrs)

### non-RFa Inference Pipeline

This process assumes that you have already trained a dataset-specific or generalized model, and just runs through the inference pipeline. I haven’t ran a video through this version of the pipeline, so there might still be bugs, but they will be very quick to solve

1. Chunk a video into sections for easier loading and processing. I used 2000 frames.  
2. Open **eulerGPU-channel.ipynb**  
   1. Update `tiff_folder` to match your dataset[^2]  
   2. Adjust `batch_size` to be the largest value that will fit on your GPU. When trying to figure this out, try powers of 2, since they’ll be very slightly faster than other numbers.   
   3. Depending on how large of a batch size you can fit, set `num_frames` to either the same value as `batch_size` or a multiple of it. (e.g. if you have a `batch_size` of 32, you might want to set num\_frames to 64, although to be honest, given that the transformation between the frames should be consistent, there shouldn’t be a large need for many frames.)  
   4. Make sure the numbers for `red_channel` and `green_channel` are correct. (remember they’re 0-indexed)  
   5. Check the range of values in your dataset and update `cutoff` as necessary. It should be a value that is slightly higher than the highest green channel value in most of the neurons. (e.g. if you have a dataset with the brightest frame’s values mostly between 200-230 then set it to 250). This value really shouldn’t matter unless some dataset randomly has a much higher or lower baseline, and it either cuts off detail or allows a massive outlier to stay in  
   6. Run all. This shouldn’t take long, although it’s entirely proportional to what was specified for `num_frames`. Check the the files selected are the correct ones  
   7. Manually check results. If you thought that the dataset had really bad channel misalignment before and it wasn’t fixed, you might want to expand the search range by changing the limits on `ALIGN_XY_RANGE` and `ALIGN_TH_RANGE`  in the 7th cell. You’ll likely want to increase the density to account for the wider range, so remember this is a cubically increasing search space. Basically if you can easily see that it’s out of alignment, you might want to boost the range since right now it’s more focused on sub-pixel stuff.  
3. Open **eulerGPU-rigalign.ipynb**  
   1. Update the `red_channel` and `BATCH_SIZE` variables in the first cell to match your setup. The “red\_channel” doesn’t have to be literally red, but is the channel that is constant that we want to use for alignment. Note it is zero-indexed, so the first channel is channel 0\.  
   2. Update the `tiff_folder` path in the second cell to match the location of your chunked video. Double check the files it prints out are correct and in the correct order.  
   3. Run All Cells \- (this can take several hours)  
   4. Evaluate the output. If it’s poorly aligned, try to figure out which frame caused it to get out of sync. If there’s a lot of movement on that frame, increase the search range values or uncommenting the expanded search snippet in the main cell  
4. Open **alignVideoSecondChannelOneTIF.ipynb**  
   1. Update `batch_size` to account for what you can fit on your GPU (although inference is less memory-hungry, it might be best to just use what you used in training)  
   2. Update `tiff_folder` in the second cell to match where your rigid-aligned video is. This should be the same as the folder in part 3b. After running this cell,, double check the files match the files created from step 3\. If you want the output from this cell saved somewhere else, you can modify `out_folder` although I’ve found it easier to just keep everything in the same folder  
   3. Update `checkpoint_path` to match your best checkpoint from training. The path should end with “ckpt-000” where 000 is whatever epoch was best. Don’t include an extension  
   4. Set `max_frames` to the size of your chunk (e.g. 2000\)  
   5. Make sure `red_chan` is correct as mentioned previously (0 \= 1st channel, 1 \= 2nd, etc.)  
   6. Check that `side_len` matches your crop size. If you have non-square images, then you’ll have to modify my code. You might also want to check beforehand that none of the zeros added during rigid-alignment are included in the crop, since that will mess up the scaling. If that’s the case and you can’t re-crop, try subtracting the median and then taking the max of that array and 0\.  
   7. If your image is double padded in the default way, set `padding` to 

      `[[0,0],[0,0],[0,0],[1,1]]`

      Else, if it’s single padded in the default way, set it to 

      `[[0,0],[0,0],[0,0],[1,0]]`

      And set `z_depth` to 2 or 3 depending on padding

   8. `log_dir` needs to be a valid path, but don’t worry about it too much  
   9. Run all cells (depending on your system’s read-write speed and load, I’ve had this take anywhere from 20min to 3+ hrs)

[^1]:  The difference between the RFa version and the non-RFa version is that the RFa version includes my naive segmentation implementation, which doesn’t work on non-RFa strains. 

[^2]:  The script is currently set up to just run on the first dataset in the list. You can manually make sure the generated transformation are consistent across the files or I can write that in, I’ve just found that loading the files is a pretty significant part of the time and never noticed any big changes in the results it generated