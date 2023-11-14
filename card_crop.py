
import cv2
import numpy as np
import math
from ultralytics import YOLO

# load card segmentation model
card_model = YOLO('./models/card_segmentation.pt')
id_model = YOLO('./models/id_extract.pt')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def get_card_vertices(src, model):
    """
    The function get the vertices of the card in the image ordered in clockwise order.

    Parameters:
        src (MatLike): The src image.
        
    Returns:
        vertices : the vertices of the card ordered in clockwise order,  in case of there is no card the output will be (None).

    """
    
    ordered_corners = None
    
    results = model(src)

    # print(results)

    if results[0].masks is None:
        return ordered_corners

    mask = results[0].masks.data[0]
    # Convert to binary for segmentation
    binary_mask = (mask.numpy() > 0.5).astype(np.uint8) * 255

    # Resize the binary mask to match the original image dimensions
    binary_mask = cv2.resize(binary_mask, (src.shape[1], src.shape[0]))

    # add extra area to contour
    binary_mask = cv2.dilate(binary_mask, np.ones((10, 10), np.uint8), iterations=1)

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


def crop_vertices(src, vertices, out_size):
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
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i + 1) % 4]
        side_length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        side_lengths.append(side_length)

        # Find the index of the longest side
    longest_side_index = side_lengths.index(max(side_lengths))
  
    reordered_vertices = np.roll(vertices, -longest_side_index, axis=0)

    # Perspective RATIO MODIFYING

    dst_corners = np.array([[0, 0], [out_size[0] - 1, 0], [out_size[0] - 1, out_size[1] - 1], [0, out_size[1] - 1]], dtype='float32')

    try:

        # Calculate the perspective transform matrix
        M = cv2.getPerspectiveTransform(reordered_vertices.astype('float32'), dst_corners)

        # Apply the perspective transformation
        result = cv2.warpPerspective(src, M, out_size)

        return result
    
    except:
        return None




def crop_card(src):
    """
    This function takes image for front or back side of national id card then return with cropped national id card with fixed size.

    Parameters:
        src (MatLike): The src image.
        
    Returns:
        card (MatLike) : The cropped card image, in case of there is no card the output will be (None).    
    """

    # get the coordinates of the quadrilateral card card_vertices
    card_vertices = get_card_vertices(src, model= card_model)
    if card_vertices is None:
        return None

    # crop card and put it in the standard size
    card_img = crop_vertices(src, card_vertices, out_size= (840, 530))

    return card_img


def crop_id (img):
        
    # Load the image you want to run detection on
    image = cv2.imread(img)
    # image = crop_card(image)
    # cv2.imshow('Resized Image', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # Perform inference
    results = id_model(image)

    # The results object is a list with a Results object.
    result = results[0]

    # Get the bounding box tensor
    bbox_tensor = result.boxes.xyxy[0]

    # Check if bbox_tensor is indeed a 1D tensor with four values
    if bbox_tensor.ndim == 1 and bbox_tensor.shape[0] == 4:
        x1, y1, x2, y2 = bbox_tensor.cpu().numpy()  # Unpack the bounding box coordinates

        # Crop the image within the bounding box
        cropped_image = image[int(y1):int(y2), int(x1):int(x2)]

        # Display the resized image
        
        return cropped_image
    else:
        print("Unexpected bbox_tensor dimensions:", bbox_tensor.shape)





