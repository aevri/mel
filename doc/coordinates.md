Conventions for representing co-ordinates
=========================================

Things can occasionally get confusing when operating on images with OpenCV from
Python.

This is due to images being accessed in a row/column fashion, whilst image
operations are typically specified in a x/y fashion.

Images are loaded with the pixels in (blue, green, red) order.

The mel project uses OpenCV's conventions, which are documented here for
reference as it can be a surprise initially.

Examples
--------

To load an image with OpenCV and get dimensions:

    >>> image = cv2.imread(image_path)
    >>> height, width, num_channels = image.shape

Note that 'height' is the first dimension.

To draw a white line across the top of the image, assuming a colour image:

    >>> image[0] = (255, 255, 255)

Note that the first dimension is again the 'height' or image row.

To draw a white line down the left of the image, assuming a colour image:

    >>> image[:,0] = (255, 255, 255)

Note that the second dimension is again the 'width' or image column.

To draw a blue line down the left of the image, assuming a colour image:

    >>> image[:,0] = (255, 0, 0)

To draw a filled red circle halfway across the top, assuming a colour image:

    >>> cv2.circle(image, (width // 2, 0), 10, (0, 0, 255), -1)

Note that the location is specified as (width, height) offset or (x, y).

To save out the image and prove this to ourselves:

    >>> imwrite(new_image_path, image)

So you can see the inconsistency between the call to cv2.circle() and the
slicing of the image with square brackets '[]'.
