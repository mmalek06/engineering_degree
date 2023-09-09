<h3>Table of contents</h3>
<br />
Different phases of work require different code, so I've splitted them into folders:

<h4>Attack plan</h4>

1. EDA
2. Establishing simple, baseline model - KNN, XGBoost?
3. Justifying the use of InceptionResNetV2 architecture as the basic choice while describing the history and architecture
4. Ignoring the dataset imbalance, searching for a non-tuned model able to beat the baseline
5. Acting on the imbalance, trying to apply techniques covered here: https://www.tensorflow.org/tutorials/structured_data/imbalanced_data
6. After optimal Inception model found, move to EfficientNet, again describing history, architecture
7. Find a ROI detection net to find ROIs on new images. Use it for the testing part:
   - feed it new images, let it find a ROI
   - cut out the interesting part based on ROI - it should be done with some margins
   - pass the cut-out ROI image to the EfficientNet for classification
8. Creating an Azure-based web api that will demonstrate how the trained networks could be used
9. Hypothesizing on further model improvements:
   - adding another model consisting of Dense layers - one that would analyze survey responses
   - merging both models, then doing the predictions

<h4>Folder contents</h4>

1. image_manipulation/resize_ham10000 - used for resizing the images coming from the HAM10000 dataset to a given size
2. image_manipulation/resize_isic2019 - same as the above but for the extended dataset
3. image_manipulation/resize_unknown - same as the above but for the unknown
4. image_manipulation/standardize_and_save_samples - performs samplewise centering and normalization - only for 
   experiments and those images won't be used in the end
5. image_manipulation/box_lesions + box_augmented_lesions - both notebooks are used to draw boxes around lesions - 
   only for debugging purposes;
6. image_manipulation/move_and_split - this one is for moving images around and splitting them into
   training and validation sets. It's also only for the augmented images, so not really used;
7. image_manipulation/categorize_images + categorize_extended_images - since the original datasets were not split
   into categories, using the csv files attached to the dataset allowed me to properly label the images, so that
   they are easily feedable to keras machinery
8. functions/* - various functions used in the notebooks
9. roi/* - region of interest detection networks
10. classifiers/* - many variations on the classification problem; the contents of this folder subfolders are run by the 
    run_long.py script

<h4>Checklist</h4>

1. done
2. done
3. todo
4. done
5. todo
6. todo
7. done - needs reiteration
8. todo
9. todo