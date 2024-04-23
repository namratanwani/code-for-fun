
import cv2
import argparse
from scipy.io import loadmat, savemat
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import math
import os



def load_as_double(filename):

    # read image
    """
    reads an image file, converts it to RGB format, grayscale, and then converts
    the grayscale image to a floating-point number.

    Args:
        filename (str): image file to be read and processed by the `load_as_double()`
            function.

    Returns:
        float: a tuple of two images: an RGB image and a grayscale image.

    """
    img = cv2.imread(filename)
    # convert image to rgb format
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # convert image to gray 
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # convert gray image to float
    img_gray = np.float64(img_gray)
    return img_rgb, img_gray

def convert_image_format_to_uint8(img):
   
    """
    converts an image from any format to a uint8 representation using OpenCV's
    `convertScaleAbs` method.

    Args:
        img (ndarray.): 3D image data to be converted into uint8 format.
            
            		- `cv2.convertScaleAbs(img)` is applied to `img`, which represents
            an image object in a particular format (e.g., BGR). This conversion
            scales and normalizes the image's pixel values to create a representation
            that can be used by the subsequent code.

    Returns:
        int: a converted image in uint8 format.

    """
    img_uint8 = cv2.convertScaleAbs(img)
    return img_uint8

def remove_noise(img):
    
    """
    blurs an image using the `medianBlur` method from the OpenCV library with a
    kernel size of 5, resulting in a noiseless image.

    Args:
        img (2D array of pixels representing a picture or image.): 2D image that
            is to be processed by the `remove_noise()` function.
            
            		- `cv2.medianBlur(img, 5)`: This line applies the median blur filter
            to `img` with a kernel size of 5 pixels. The returned image is stored
            in `img_noiseless`.

    Returns:
        image.: a noiseless image obtained through median blurring with a kernel
        size of 5 pixels.
        
        		- `img_noiseless`: The output image after applying the median blur filter
        with a size of 5 pixels. This is a numpy array representing the noiseless
        image.

    """
    img_noiseless = cv2.medianBlur(img,5)
    return img_noiseless

def find_coordinates_of_circles(img, minRadius = 24, maxRadius = 25):
    
    """
    detects circles in an image using the Hough transform and returns the coordinates
    of the detected circles and their radii.

    Args:
        img (ndarray (2D array).): 2D image that the Hough Circle Transform is
            applied to in order to detect circles within the image.
            
            		- `img`: This is an image captured by a camera or loaded from a file.
            It is a numpy array with dimensions (height, width, channels), where
            each channel represents a different color component (e.g., RGB).
            		- `minRadius`: This is an integer parameter that sets the minimum
            radius of a circle in pixels for it to be considered a circle. The
            default value is 24 pixels.
            		- `maxRadius`: This is also an integer parameter that sets the maximum
            radius of a circle in pixels for it to be considered a circle. The
            default value is 25 pixels.
            		- `cv2.HoughCircles()`: This is a function from the OpenCV library
            that takes as input an image and returns a list of circles detected
            in the image using the Hough transform. The function takes several
            parameters, including `dp`, `minDist`, `param1`, `param2`, `minRadius`,
            and `maxRadius`. These parameters control various aspects of the circle
            detection, such as the resolution of the circles, the minimum distance
            between two circles, and the range of radii for the circles.
        minRadius (int): minimum diameter of circles to be detected and drawn on
            the image.
        maxRadius (int): maximum radius of a circle that can be detected in the
            image using the Hough transformation, and it is used to filter out
            circles with larger radii than the specified value.

    Returns:
        `numpy.array`.: a list of circle coordinates and their radii, obtained
        through Hough transform.
        
        		- `coords`: A list of tuples containing the coordinates of the circles
        detected in the image. Each tuple contains the x and y coordinates of a circle.
        		- `r`: The radius of each circle.
        
        	Therefore, the output of the function can be described as a list of circle
        coordinates with their corresponding radii.

    """
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, dp=3, minDist=120, param1=150, param2=50, minRadius=minRadius, maxRadius=maxRadius)
    circles = np.round(circles[0, :]).astype(int)
    coords = []
    for (x, y, r) in circles:
            # print(x,y)
            cv2.circle(img, (x, y), r, (255, 0, 0), 2)
            # append circle coordinates
            coords.append((x,y))
    return coords,r

def unskew_image(gray_img, color_img, circle_coords):
    
    
    """
    applies a perspective transform to an image given a set of corners representing
    a circle. The transformed image is returned as both grayscale and RGB images.

    Args:
        gray_img (float): 8-bit grayscale image that is transformed using the
            perspective transform.
        color_img (ndarray object of shape (height, width, 3), where height and
            width denote the dimensions of the image in pixels, and 3 represents
            the number of color channels (red, green, blue) in each pixel.):
            3-channel RGB image that is transformed by the perspective transform.
            
            		- `color_img` is an image object, which means it has several properties
            and attributes related to its pixel data and structure.
            		- `color_img.shape` returns the size of the image in pixels (e.g.,
            `(100, 100)`).
            		- `color_img.dtype` returns the data type of the image pixels
            (`np.uint8` by default).
            		- `color_img.ndim` returns the number of dimensions in the image
            array (`2` for a standard RGB image).
            		- `color_img.tolist()` converts the image pixels to a list, which
            can be used for further processing or visualization.
        circle_coords (float): 4 corners of a circle in the image, which are used
            to calculate the perspective transform matrix and apply it to the image.

    Returns:
        float: a pair of images, one grayscale and one color, that have been
        perspective-transformed based on a set of input coordinates.

    """
    corner1 = np.array(circle_coords[0])
    corner2 = np.array([circle_coords[2][0],circle_coords[2][1]]) 
    corner3 = np.array(circle_coords[3])
    corner4 = np.array([circle_coords[1][0],circle_coords[1][1]])
    w = 500
    h = 500
    # Define the source and destination points
    src_points = np.array([corner1, corner2, corner3, corner4], dtype=np.float32)
    dst_points = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)

    # Calculate the perspective transform matrix and apply it to the image
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    warped_rgb = cv2.warpPerspective(color_img, M, (w, h))
    warped_gray = cv2.warpPerspective(gray_img, M, (w, h))
    return warped_gray, warped_rgb

def draw_rectangle(img, coords, r):
   
    """
    takes an image (`img`), coordinates of a rectangle to draw (`coords`), and a
    radius for the rectangle (`r`). It draws a green rectangle on the image using
    OpenCV's `rectangle` method.

    Args:
        img (image.): 2D image to which a rectangular shape will be drawn using
            the `rectangle()` function from OpenCV.
            
            		- `img`: This is an image object that contains an array of pixels
            in a 2D matrix format.
            		- `coords`: An array of coordinates representing the rectangle to
            be drawn on the image. The coordinates are in (x, y) format, where x
            and y are integers representing the pixel positions within the image.
            		- `r`: A single integer value representing the radius of the rectangle.
        coords (list): 2D coordinates of the top-left corner of the rectangle to
            be drawn on the given image.
        r (int): thickness of the rectangle that is being drawn on the image.

    Returns:
        image.: a rectangle drawn on the input image using the specified coordinates
        and radius.
        
        		- The `rectangleDrawn` variable is a 2D array of shape (height, width,
        3) representing the drawn rectangle on the input image `img`.
        		- Each pixel in the array corresponds to a point (x, y) in the rectangle
        and has a color value (0, 0, 255) indicating the color of the rectangle.
        		- The size of the rectangle is specified by the `r` parameter, which
        represents the width of the rectangle.

    """
    left_min = min(coords[1])
    right_max = max(coords[1])
    # print(left_min, right_max)
    rectangleDrawn = cv2.rectangle(img, (left_min+r, left_min+r), (right_max-r, right_max-r), (0, 0, 255), 2)
    return rectangleDrawn

def crop_image(img, coords, r):
   
    """
    crops an image based on given coordinates and a ratio. It returns the cropped
    image.

    Args:
        img (image.): 2D image to be cropped.
            
            		- `img`: The original image to be cropped. It is an array of pixels
            with shape `(height, width, channels)`, where `height`, `width`, and
            `channels` represent the height, width, and number of color channels
            (either 3 for RGB or 4 for RGBA), respectively.
        coords (list): 2D coordinates of the region of interest (ROI) within the
            original image that should be cropped.
        r (int): margin or border to be cropped from the image.

    Returns:
        `image`.: a cropped version of the original image.
        
        	1/ The cropped image is a NumPy array representing the portion of the
        original image that falls within the specified coordinates.
        	2/ The shape of the cropped image is (height, width), where height and
        width are the dimensions of the original image.
        	3/ The pixels in the cropped image have values that are obtained by taking
        the corresponding pixels from the original image and shifting them
        horizontally by the specified `r` value.

    """
    left_min = min(coords[1])
    right_max = max(coords[1])
    cropped_image = img[left_min+r: right_max-r, left_min+r: right_max-r]
    return cropped_image

def find_edges(img):
    
    # The values 5 and 75 are threshold values used to determine which pixels are considered edges. 
    # Pixels with gradient values higher than the 75 threshold are considered edges, while pixels 
    # with gradient values lower than the 5 threshold are not considered edges. The edges variable
    # stores the resulting edge map.
    """
    performs edge detection on an image using Canny edge detection, followed by
    dilation and contour detection to identify squares in the dilated edges map.
    It returns the dilated edges and unique square coordinates.

    Args:
        img (ndarray or NumPy array.): 2D image that undergoes edge detection and
            contour identification.
            
            		- `img`: The input image, which is a 2D array of grayscale pixels
            with dimensions `(height, width)`.
            		- `cv2.Canny()`: Applies the Canny edge detection algorithm to `img`,
            using the threshold values of 5 and 75 for determining edges. The
            resulting edge map is stored in `edges`.
            		- `cv2.getStructuringElement()`: Returns a rectangular kernel with
            dimensions `(2, 2)`, which is used for dilation in the next step.
            		- `cv2.dilate()`: Applies dilation to the edge map using the provided
            kernel and number of iterations. The resulting dilated edge map is
            stored in `dilated_edges`.
            		- `cv2.findContours()`: Identifies contours (shapes) in the dilated
            edge map. The contour coordinates are stored in `contours`.
            		- `approxPolyDP()`: Approximates the contour as a polygon using the
            Donath-Leporino algorithm, and returns the polygon vertices as an array.

    Returns:
        list: a list of unique square coordinates and the dilated edge map.

    """
    edges = cv2.Canny(img, 5, 75)
    # creates a 2x2 rectangular kernel that is used for dilation, which helps 
    # connect and smooth out the edges.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    # uses the dilation operation to thicken the edges of the image. The dilate function takes the edge
    # map edges, the kernel kernel, and the number of iterations to perform the dilation. The resulting
    # dilated edge map is stored in dilated_edges.
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)
    # identify the contours (shapes) of the objects in the dilated edge map. 
    contours, _ = cv2.findContours(dilated_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    lst = []
    visited = []
    square_coords = []
    for cnt in contours:
        # Approximate contour as a polygon
        approx = cv2.approxPolyDP(cnt, 0.1*cv2.arcLength(cnt, True), True)

        # If polygon has 4 sides, it's a square (in this case)
        if len(approx) == 4:
            # Draw bounding rectangle
            x, y, w, h = cv2.boundingRect(approx)
            area = w * h
            # Check if square is within desired size range
            if 5800 < area < 6600:
                # Retrieve coordinates of square vertices
                square_coords.append((x, y))

    # initialise empty dictionary
    d = {}
    # loop through square coordinates 
    for i in range(0,len(square_coords)-1):
        # get difference between consecutive coordinates
        ans = np.array(square_coords[i]) - np.array(square_coords[i+1])
        # get absolute sum of difference between coordinates
        s = abs(sum(ans))
        # if absolute value is less that 12 then coordinates point to the same square and are close to each other
        if s < 12:
            if i not in d.keys():
                d[i] = square_coords[i]
            if i+1 not in d.keys():
                d[i+1] = square_coords[i]
        # if absolute value is more than 20 then coordinates point to different squares
        elif s > 20:
            if i not in d.keys():
                d[i] = square_coords[i]
            if i+1 not in d.keys():
                d[i+1] = square_coords[i+1]
    # the dictionary still might contain some repitive values
    # this is finally removed by the following code
    final_coords = []
    # iterate through dictionary's values
    for val in d.values():
        # append values if they are not added to final_coords list
        if val not in final_coords:
            final_coords.append(val)

    # print(square_coords)
    #return dilated edges and unique square coordinates
    return dilated_edges, final_coords

def map_coords_to_sections(coords):
   
    # initialise a dictionary with each key mapping 
    # to row/column number of the square. Here, each key 
    # is a pixel value. 
    
    """
    maps pixel coordinates to rows and columns within a square grid, using a
    dictionary to translate coordinates to numerical indices.

    Args:
        coords (list): 2D coordinates of a grid, which are used to map each
            coordinate to a specific row and column number in a nested list matrix.

    Returns:
        list: a nested list of tuples representing a grid of coordinates and their
        corresponding sections.

    """
    mapped = {
        23: 0,
        108: 1,
        193: 2,
        278: 3
    }

    # initialise a nested list with tuple (0,0)
    matrix = [[(0,0) for _ in range(4)] for _ in range(4)]
    # iterate through coordinates
    for tup in coords:
       
        x = tup[0]
        y = tup[1]

        for key in mapped.keys():
            
            result_x = math.isclose(x, key, abs_tol = 10)
            if result_x:
                
                mat_x = mapped[key]
             
            result_y = math.isclose(y, key, abs_tol = 10)
            if result_y:
               
                mat_y = mapped[key]
       
        matrix[mat_y][mat_x] = (x,y)
   
    return matrix

def get_color_matrix(img, matrix):



    """
    takes an image and a matrix as input, generates a color matrix for each pixel
    in the image based on the values in the matrix, and returns the resulting color
    matrix.

    Args:
        img (ndarray object.): 2D image that is processed and analyzed to generate
            the color matrix.
            
            		- `img` is an 8-bit grayscale image, represented as a numpy array.
            		- The shape of `img` is `(rows, cols)`, where `rows` and `cols` are
            the dimensions of the image.
            		- Each element of `img` is a value between 0 and 255, representing
            the intensity of the pixel at that position.
            		- The image may have various properties such as contrast, brightness,
            and saturation, which can affect the resulting color matrix.
        matrix (float): 2D array of pixel values from the image, which is used to
            compute the mean and binary thresholding for color classification.

    Returns:
        str: a 4x4 matrix of color labels for each pixel in an image, based on the
        intensity values of neighboring pixels.

    """
    colors = [[None for _ in range(4)] for _ in range(4)]

    for i in range(4):
        for j in range(4):
           
            l = matrix[i][j]
         
            if l == (0,0):
               
                colors[i][j] = 'undetected'
                
                continue
            
            section = img[l[1]:l[1]+80, l[0]:l[0]+80]
            
            section_mean = cv2.mean(section)
           
            binary_list = [1 if val > 130 else 0 for val in section_mean[:3] ]
           
            r = binary_list[0]
            g = binary_list[1]
            b = binary_list[2]
            
            
            if r and g and b:
                colors[i][j] = 'white'
            elif r and g :
                colors[i][j] = 'yellow'
            elif r and b:
                colors[i][j] = 'purple'
            elif b:
                colors[i][j] = 'blue'
            elif r:
                colors[i][j] = 'red'
            elif g:
                colors[i][j] = 'green'
            else:
                colors[i][j] = 'unknown color'

 
    return colors

# get all files to test
files = []
for i in range(1,6):
    files.append('org_'+str(i))
for i in range(1,6):
    files.append('noise_'+str(i))
files


path = 'masters/CV/images2/images/' 
loc = 1
fig=plt.figure(figsize=(50, 90))

for file in files:
   

    filepath = path + file + '.png' 
  
  
    rgb_img, gray_img = load_as_double(filepath)
  
    gray_img_uint8 = convert_image_format_to_uint8(gray_img)
    
    rgb_img_noiseless = remove_noise(rgb_img)
    
    gray_img_noiseless = remove_noise(gray_img_uint8)
    
    circle_coordinates, radius = find_coordinates_of_circles(gray_img_noiseless)
   
    rectangle = draw_rectangle(gray_img_noiseless, circle_coordinates, radius)
   
    cropped_image_rgb = crop_image(rgb_img_noiseless,circle_coordinates, radius)
    
    cropped_image_gray = crop_image(gray_img_noiseless,circle_coordinates, radius)
   
    cropped_image_gray_img_uint8 = convert_image_format_to_uint8(cropped_image_gray)
    
    cropped_image_gray_noiseless_1 = remove_noise(cropped_image_gray_img_uint8)
    cropped_image_gray_noiseless_2 = remove_noise(cropped_image_gray_noiseless_1)
    
    dilated_edges, square_coordinates = find_edges(cropped_image_gray_noiseless_2)
    
    mapped_matrix = map_coords_to_sections(square_coordinates)
   
    final_matrix = get_color_matrix(cropped_image_rgb, mapped_matrix)
    
    fig.add_subplot(10,1 , loc)
    plt.imshow(cropped_image_rgb)
    plt.title(file+'.png' + ' - '+ str(final_matrix))
    print(file+'.png' + ' - '+ str(final_matrix))
    loc += 1

    # store result in answers directory inside 'path' 
    dirpath = path + 'answers/'
    if not os.path.exists(dirpath):
        os.mkdir(dirpath)
    savemat(dirpath + file +'.mat', mdict = {'res': final_matrix})
print("Answers stored in .mat format inside 'answers' folder inside", path)

# get all files to test
files_skewed = []
for i in range(1,6):
    files_skewed.append('rot_'+str(i))
for i in range(1,6):
    files_skewed.append('proj_'+str(i))
files_skewed

# The following code is not part of the pipeline and hence, has been coded separately
# to unskew rotated images
path = 'masters/CV/images2/images/'
loc = 1
fig=plt.figure(figsize=(50, 90))
for file in files_skewed:
    # get file path
    filepath = path + file + '.png' 
    # print(filepath)
    rgb_img, gray_img = load_as_double(filepath)
    # convert image to uint8 format
    gray_img_uint8 = convert_image_format_to_uint8(gray_img)
    # remove noise from color image
    rgb_img_noiseless = remove_noise(rgb_img)
    # remove noise from gray image
    gray_img_noiseless = remove_noise(gray_img_uint8)
    # get corner circle coordinates with their radii
    try:
        circle_coordinates, radius = find_coordinates_of_circles(gray_img_noiseless, 24,26)
        undistorted_image_gray, undistorted_image_rgb = unskew_image(gray_img_noiseless, rgb_img_noiseless, circle_coordinates)
        fig.add_subplot(10,1 , loc)
        plt.title(file+'.jpg')
        plt.imshow(undistorted_image_gray)
        loc += 1 
    except:
        pass