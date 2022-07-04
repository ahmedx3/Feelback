## TODO
- Try EBIF for age

- Try Gabor filters used in the emotions

- Create the loader for the AGFW dataset

- Find out how AGFW cropped the faces

- Check new classifiers and grid search for them to find best parameters

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
- Change the Age problem to regression
- Age regression grid search for SVR to find best parameters
- Integrate the Age in the main pipeline
- Mix the datasets
- Create a model for Age
