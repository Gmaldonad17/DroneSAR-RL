import cv2
import numpy as np
from custom_environment import landscapev0

def main():
    # Create an instance of the landscape_map class
    pixels_per_meter = 9
    landscape_size = 150
    
    map_generator = landscapev0(pixels_per_meter, landscape_size)

    # Display the generated map using OpenCV
    cv2.imwrite('generated_map.png', map_generator.img_map)
    cv2.imshow("Generated Landscape Map", map_generator.img_map/255)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()