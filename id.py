from ultralytics import YOLO
from card_crop import crop_id
import cv2

model = YOLO(r"models\final_ocr_front_back.pt")
    
    
def read_image(img):
    """
    This function take back card image and get national number on the card.
    Parameters:
        src (MatLike)
    Returns:
        data (str)
    """
    imge_back = crop_id(img)
    
    if imge_back is None:
        return {"status": "failed"}
    # Crop the image    
    # Set the confidence threshold for the model to 0.25
    model.conf = 0.25  # Adjust this line according to the actual API
    # Run inference on the cropped image
    results = model(imge_back)

    # Extract the bounding boxes and their corresponding classes
    boxes = results[0].boxes.xywh  # Assuming the boxes are in xywh format
    classes = results[0].boxes.cls

    # Create a list of tuples, where each tuple contains (x-coordinate, class_name)
    x_coords_with_classes = [
        (box[0], model.names[int(c)]) for box, c in zip(boxes, classes)
    ]

    # Sort this list based on the x-coordinate
    sorted_classes = [
        cls for _, cls in sorted(x_coords_with_classes, key=lambda x: x[0])
    ]

    # Concatenate the sorted class names into a single string
    id_num_str = "".join(sorted_classes)
    result = 'failed' if len(id_num_str) != 14 else id_num_str
    
    return result
# image = r"C:\Users\moham\Downloads\WhatsApp Image 2023-11-14 at 12.02.53 AM.jpeg "
# print(read_image(image)) 

