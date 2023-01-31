# Importing all necessary libraries
import cv2
import os
import numpy as np
import math
  
def resize_linear(image_matrix, new_height:int, new_width:int):
    """Perform a pure-numpy linear-resampled resize of an image."""
    output_image = np.zeros((new_height, new_width), dtype=image_matrix.dtype)
    original_height, original_width = image_matrix.shape
    inv_scale_factor_y = original_height/new_height
    inv_scale_factor_x = original_width/new_width

    # Serial operation
    for new_y in range(new_height):
        for new_x in range(new_width):
            # If you had a color image, you could repeat this with all channels here.
            # Find sub-pixels data:
            old_x = new_x * inv_scale_factor_x
            old_y = new_y * inv_scale_factor_y
            x_fraction = old_x - math.floor(old_x)
            y_fraction = old_y - math.floor(old_y)

            # Sample four neighboring pixels:
            left_upper = image_matrix[math.floor(old_y), math.floor(old_x)]
            right_upper = image_matrix[math.floor(old_y), min(image_matrix.shape[1] - 1, math.ceil(old_x))]
            left_lower = image_matrix[min(image_matrix.shape[0] - 1, math.ceil(old_y)), math.floor(old_x)]
            right_lower = image_matrix[min(image_matrix.shape[0] - 1, math.ceil(old_y)), min(image_matrix.shape[1] - 1, math.ceil(old_x))]

            # Interpolate horizontally:
            blend_top = (right_upper * x_fraction) + (left_upper * (1.0 - x_fraction))
            blend_bottom = (right_lower * x_fraction) + (left_lower * (1.0 - x_fraction))
            # Interpolate vertically:
            final_blend = (blend_top * y_fraction) + (blend_bottom * (1.0 - y_fraction))
            output_image[new_y, new_x] = final_blend

    return output_image  

# Read the video from specified path
cam = cv2.VideoCapture("DIW_Videos/Tip_Videos/DIW_T_5.mp4")

try:
    # creating a folder named data
    if not os.path.exists('data'):
        os.makedirs('data')
  
# if not created then raise error
except OSError:
    print ('Error: Creating directory of data')
  
# frame
currentframe = 0
prevLandmarks = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
while(True):
      
    # reading from frame
    ret,frame = cam.read()
    # Crops image to 1070x1070
    frame = np.delete(frame, slice(560, 852), 1)
    frame = np.delete(frame, slice(0, 80), 1)
    #frame = np.delete(frame, slice(0, frame.shape[0]-1070), 0)
    #frame = np.delete(frame, slice(480, 640), 1)
    # Make frame black and white
    frame = np.dot(frame[...,:3], [0.299, 0.587, 0.114])
    
    # Scales image down to generic 480x480
  
    
    newLandmarks = np.array([frame[140][140], frame[240][140], frame[340][140], frame[140][240], 
                             frame[240][240], frame[340][240], frame[140][340], frame[240][340], frame[340][340]])
     
    if(np.sum(abs(prevLandmarks-newLandmarks)) < 35):
        prevLandmarks = newLandmarks
        continue
    
    if ret:
        # if video is still left continue creating images
        name = './data/frame' + str(currentframe) + '.jpg'
        print ('Creating...' + name)
    
        # writing the extracted images
        cv2.imwrite(name, frame)
        prevLandmarks = newLandmarks
        # increasing counter so that it will
        # show how many frames are created
        currentframe += 1
    else:
        break
  
# Release all space and windows once done
cam.release()
cv2.destroyAllWindows()