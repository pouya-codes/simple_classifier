import openslide
import os
import numpy as np
from PIL import Image
import re
from matplotlib.path import Path
import glob
import concurrent.futures


# Define the path to the slides and annotations
slides_path = '/home/pouya/Develop/Sayna/Codes/Data/Slides'
annotation_dir = '/home/pouya/Develop/Sayna/Codes/Data/Annotations'
output_dir = '/home/pouya/Develop/Sayna/Codes/Data/Patches'
patch_size = (1024, 1024)
resize_size = (256, 256)

max_patches_per_area = 50

os.makedirs(output_dir, exist_ok=True)



import concurrent.futures

def process_slide(slide_path, annotation_path):
    # Open the slide
    print('Processing:', slide_path)
    slide = openslide.OpenSlide(slide_path)
    case_id = os.path.basename(slide_path)[:-4]
    # Process each annotation
    with open(annotation_path, 'r') as f:
        annotations = f.readlines()
    for annotation in annotations:
        # Split the annotation by the label
        parts = re.split(r'(Tumor|Stroma)', annotation)

        # Process each polygon
        for i in range(1, len(parts), 2):
            label = parts[i]
            
            points_str = parts[i + 1].strip('[]')
            points = re.findall(r'Point: (.*?), (.*?)(?:,|$)', points_str)
            # Convert the points to integers
            points = [(int(float(x.replace(']',''))), int(float(y.replace(']','')))) for x, y in points]
            # Create a path from the points
            path = Path(points)

            # Calculate the bounding box of the polygon
            min_x = min(x for x, y in points)
            max_x = max(x for x, y in points)
            min_y = min(y for x, y in points)
            max_y = max(y for x, y in points)

            # patch = slide.read_region((min_x, min_y), 0, (max_x - min_x, max_y - min_y))
            # resized_img = patch.resize(((max_x - min_x)//10, ((max_y - min_y))//10))
            # os.makedirs(os.path.join(output_dir, label, case_id), exist_ok=True)
            # resized_img.save(os.path.join(output_dir, label, case_id, f'polygon_{i}_{label}_{min_x}_{min_y}.png'))
            
            counter = 0
            # Iterate over the pixels in the bounding box
            for x in range(min_x, max_x, patch_size[0]):
                for y in range(min_y, max_y, patch_size[1]):
                    # Create a grid of points in the patch
                    patch_points = [(x + dx, y + dy) for dx in range(patch_size[0]) for dy in range(patch_size[1])]

                    # Check if all points in the patch are inside the polygon
                    if all(path.contains_point(point) for point in patch_points):
                        # Extract the patch
                        patch = slide.read_region((x, y), 0, patch_size)

                        # Convert the patch to an image
                        patch_img = Image.fromarray(np.array(patch)[:, :, :3]).resize(resize_size)

                        # Save the patch
                        os.makedirs(os.path.join(output_dir, label, case_id), exist_ok=True)
                        patch_img.save(os.path.join(output_dir, label, case_id, f'patch_{i}_{x}_{y}.png'))
                        counter += 1
                        if counter >= max_patches_per_area:
                            break

    # Close the slide
    slide.close()


            

if __name__ == "__main__":
    # Get a list of all annotation files
    annotation_files = glob.glob(os.path.join(annotation_dir, '*.txt'))

    with concurrent.futures.ThreadPoolExecutor() as executor:
        for annotation_path in annotation_files:
            case_id = os.path.basename(annotation_path)[:-4]
            slide_path = os.path.join(slides_path, case_id + '.svs')
            print(slide_path)
            if not os.path.exists(slide_path):
                print('Slide not found:', slide_path)
                continue
            else:
                executor.submit(process_slide, slide_path, annotation_path)