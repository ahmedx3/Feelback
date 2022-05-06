## TODO
- Experiment with the cropping size in the main

- Mix the datasets

- Create the loader for the datasets of Age

- Create a model for Age

- Check new classifiers and grid search for them to find best parameters

- Check more feature extraction methods (FPLBP)

- Check more preprocessing and resize images (Frontalization & facial landmark localization)

## Done
- Implement loading Datasets to have the same interface
- Create the training pipeline with svm as base
- Create the single image validation to check a single image test
- Create dataset validation to check on different datasets
- Try LPQ for training and Implement it
- Check the misclasifications and the functions to select randomly
- Visualize the datasets and check their statistics
- Check regular LBP and uniform LBP features for training
- Add preprocessing histogram equalization and resizing
- Change the training and testing datasets (Train with mixture of datasets or change order)
- K fold Validation
- Create video capture that shows label
- Train on Data that was extracted from the face detection model
- Check lighting effect and preprocessing to unify the images
- Try the LBP on regions of the image divided to 16 20x20
- Change the predicted label to be an aggregation of the prediction over the video with more than one method