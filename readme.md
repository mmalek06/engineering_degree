<h3>Table of contents</h3>
<br />
Different phases of work require different code, so I've splitted them into folders:

<h4>Image manipulation</h4>

1. image_manipulation/resizing - used for resizing the images coming from the HAM10000 dataset to a 150x200 size;
2. image_manipulation/resize_extended_dataset - same as the above but for the extended dataset;
3. image_manipulation/augmentation - I thought some augmentation upfront could be useful, but during the 
   experimentation it turned out what keras can do on the fly is enough for this POC. I'm leaving this notebook
   in case further work reveals that it's actually required;
4. image_manipulation/box_lesions + box_augmented_lesions - both notebooks are used to draw boxes around lesions - 
   only for debugging purposes;
5. image_manipulation/move_and_split - this one is for moving images around and splitting them into
   training and validation sets. It's also only for the augmented images, so not really used;
6. image_manipulation/categorize_images + categorize_extended_images - since the original datasets were not splitted
   into categories, using the csv files attached to the dataset allowed me to properly label the images, so that
   they are easily feedable to keras machinery.
7. image_manipulation/layers - in case a Dataset is used to load the images, some useful operations of
   ImageDataGenerator won't be available. Classes in this directory perform those operations on a dataset.

<h4>ROI</h4>

The WHY: I thought that when I'm in the phase of writing the application that will use the trained classifier
some Region of Interest detection neural net could come in handy - pictures will be taken from different
perspectives and maybe it will boost the user experience if the mobile app could show bounding boxes around 
lesions - that way we would increase the probability of the user actually taking a picture of the correct
spot.

1. roi/roi_detection - this notebook contains the NN that will be used ultimately if I decide to use ROI 
   detection for this poc
2. roi/roi_detection_inception_resnet_v2_200_150 - this one was trained using transfer learning based on
   the InceptionResNetV2 architecture
3. roi/roi_detection_on_augmented_imgs - roi detection on the images supplemented with static augmentation
4. roi/test_best_roi - for debugging purposes - it allowed me to see the actually detected bounding boxes

<h4>Classifiers </h4>

<TODO>
