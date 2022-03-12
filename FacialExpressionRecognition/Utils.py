import cv2

def show_image(image, title="Image"):
    """Shows an Image with a title

    Args:
        image (image): Image to show
        title (str, optional): Title of the Image. Defaults to "Image".
    """
    cv2.imshow(title, image)
    cv2.waitKey(0)