
import cv2
import numpy as np
import math
from ultralytics import YOLO

# load card segmentation model
model = YOLO(r'models\card_segmentation.pt')

# Load the Haar Cascade face detection classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def get_card_vertices(img):
    """
    The function get the vertices of the card in the image ordered in clockwise order.

    Parameters:
        src (MatLike): The src image.
        
    Returns:
        vertices : the vertices of the card ordered in clockwise order,  in case of there is no card the output will be (None).

    """
    
    ordered_corners = None
    
    results = model(img)

    if results[0].masks is None:
        return ordered_corners

    mask = results[0].masks.data[0]
    # Convert to binary for segmentation
    binary_mask = (mask.cpu().numpy() > 0.5).astype(np.uint8) * 255

    # Resize the binary mask to match the original image dimensions
    binary_mask = cv2.resize(binary_mask, (img.shape[1], img.shape[0]))

    # Find contours
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Assuming the largest contour is the ID card
    card_contour = max(contours, key=cv2.contourArea)

    # Find corners of the card
    epsilon = 0.05 * cv2.arcLength(card_contour, True)
    approx = cv2.approxPolyDP(card_contour, epsilon, True)

    # Reorder the points to ensure they are in clockwise order
    corners = np.array(approx).reshape(-1, 2)
    ordered_corners = np.zeros_like(corners)

    # Calculate the centroid of the points
    centroid = np.mean(corners, axis=0)

    # Sort the points based on their angle from the centroid
    angles = np.arctan2(corners[:, 1] - centroid[1], corners[:, 0] - centroid[0])
    sorted_indices = np.argsort(angles)

    # Reorder the corners
    for i in range(4):
        ordered_corners[i] = corners[sorted_indices[i]]

    return ordered_corners


def crop_card(img, card_vertices, out_size= (840, 530)):
    """
    This function crop card in horizontal.

    Parameters:
        src (MatLike): The src image.
        
    Returns:
        card (MatLike) : The cropped card image.
    """
    
    # Reorder the vertices so that it starts with the longest side

        # Calculate the lengths of the four sides of the quadrilateral
    side_lengths = []
    for i in range(4):
        x1, y1 = card_vertices[i]
        x2, y2 = card_vertices[(i + 1) % 4]
        side_length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        side_lengths.append(side_length)

        # Find the index of the longest side
    longest_side_index = side_lengths.index(max(side_lengths))

        # Calculate the slope angle of the longest side
    x1, y1 = card_vertices[longest_side_index]
    x2, y2 = card_vertices[(longest_side_index + 1) % 4]
  
    reordered_vertices = np.roll(card_vertices, -longest_side_index, axis=0)

    # Perspective RATIO MODIFYING

    dst_corners = np.array([[0, 0], [out_size[0] - 1, 0], [out_size[0] - 1, out_size[1] - 1], [0, out_size[1] - 1]], dtype='float32')

    # Calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(reordered_vertices.astype('float32'), dst_corners)

    # Apply the perspective transformation
    result = cv2.warpPerspective(img, M, out_size)
    
    return result




def crop_card_front(src):
    """
    This function takes image for front side of national id card then return with cropped national id card with fixed size.

    Parameters:
        src (MatLike): The src image.
        
    Returns:
        card (MatLike) : The cropped card image, in case of there is no card the output will be (None).    
    """

    # get the coordinates of the quadrilateral card card_vertices
    card_vertices = get_card_vertices(src)
    if card_vertices is None:
        return None

    # crop card and put it in the standard size
    card_img = crop_card(src, card_vertices)

    # rotate card if needed

        # Convert the image to grayscale
    gray = cv2.cvtColor(card_img, cv2.COLOR_BGR2GRAY)
    gray_rotated = cv2.rotate(gray, cv2.ROTATE_180)

        # Detect faces in the grayscale image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=20, minSize=(30, 30))
    faces_rotated = face_cascade.detectMultiScale(gray_rotated, scaleFactor=1.1, minNeighbors=20, minSize=(30, 30))
    
    if len(faces) > 0 and len(faces_rotated) == 0:
        return card_img     # Upright faces detected. There may not be inverted faces in the image

    elif len(faces) == 0 and len(faces_rotated) > 0 :
        return cv2.rotate(card_img, cv2.ROTATE_180)     # Rotate the image by 180 degrees
    
    elif len(faces) == 0 and len(faces_rotated) == 0 :
        return None
    
    else:
        # compare area
        if faces[0][2]*faces[0][3] >= faces_rotated[0][2]*faces_rotated[0][3]:
            return card_img
        
        else:
            return cv2.rotate(card_img, cv2.ROTATE_180)



def crop_card_back(src):
    """
    This function takes image for back side of national id card then return with cropped national id card with fixed size.

     Parameters:
        src (MatLike): The src image.
        
    Returns:
        card (MatLike) : The cropped card image, in case of there is no card the output will be (None).    
    """

    # get the coordinates of the quadrilateral card card_vertices
    card_vertices = get_card_vertices(src)
    if card_vertices is None:
        return None

    # crop card and put it in the standard size
    card_img = crop_card(src, card_vertices)

    

    # rotate if needed

    #********** detect barcode pos ***********

    # Convert the image to grayscale
    gray = cv2.cvtColor(card_img, cv2.COLOR_BGR2GRAY)
    # Threshold the image to create a binary mask of black pixels
    mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    # Invert the binary image
    mask = cv2.bitwise_not(mask)
    # Define a kernel (structuring element) for erosion
    kernel = np.ones((3, 3), np.uint8)
    # Perform erosion on the image
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=5)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Assuming the largest contour is the ID card
    barcode_contour = max(contours, key=cv2.contourArea)

    # Calculate the moments of the contour to find its centroid
    M = cv2.moments(barcode_contour)
    
    cy = 0

    if M["m00"] != 0:
        # Calculate the centroid coordinates (cx, cy)
        # cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

    if cy >= card_img.shape[0]/2:
        return card_img
    
    else:
        return cv2.rotate(card_img, cv2.ROTATE_180)     # Rotate the image by 180 degrees


