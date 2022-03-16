import cv2
import numpy as np
from matplotlib import pyplot as plt

def show_image(image, title="Image"):
    """Shows an Image with a title

    Args:
        image (image): Image to show
        title (str, optional): Title of the Image. Defaults to "Image".
    """
    cv2.imshow(title, image)
    cv2.waitKey(0)

def show_images(images, titles = []):
    """Shows a list of images sequentially

    Args:
        images (List of Images): Images to Show
        titles (list, optional): Titles of Images. Defaults to [].
    """
    for index in range(len(images)):
        if type(titles) == list:
            title = titles[index] if len(titles) == len(images) else "Image"
        elif type(titles) == str:
            title = titles
        show_image(images[index], title)

def get_random_incorrect_labeled(labels, predictions, image_paths, number_of_images):
    """Gets Random Images from the incorrect labeled ones

    Args:
        labels (_type_): True Labels for the images
        predictions (_type_): Predictions by the model
        image_paths (_type_): Absolute Image paths as they are on disk
        number_of_images (_type_): The number of random images to return

    Returns:
        List of Images: List of CV2 Images with the size of number_of_images
    """
    
    # Get missclassified images indices
    np_labels = np.asarray(labels)
    image_paths = np.asarray(image_paths)
    missclassified = np.nonzero(np_labels != predictions)[0]

    # Choose random indices
    index = np.random.choice(missclassified, number_of_images, replace=False)
    images = image_paths[index]

    # Initialize images array
    random_images = []

    # Read the images and return them
    for image in images:
        img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        random_images.append(img)

    return random_images
    

def plot_dataset_histogram(labels):
    """Plots a Histogram of the Dataset
    """

    # plot piechart
    plt.hist(labels, bins=len(np.unique(labels)), rwidth=0.9)
    plt.show(block=True) 

def plot_dataset_piechart(labels):
    """Plots a Pie Chart of the Dataset
    """

    # Extract Unique values and count
    labels = np.asarray(labels)
    values, counts = np.unique(labels, return_counts=True)

    # Change labels
    modified_labels = []
    for index in range(len(counts)):
        modified_labels.append(str(str(values[index]) + " : " + str(counts[index])))
    
    modified_labels = values if len(values) > 2 else modified_labels
        
    # plot piechart
    plt.pie(counts, labels = modified_labels,startangle=90)
    plt.legend()
    plt.show(block=True)

