import sys
import numpy as np
import SimpleITK as sitk

def possible_neighbours(coords, image):
    [z, y, x] = coords
    neighbours = []
    if z > 0:
        neighbours.append([z-1, y, x])
    if z < image.shape[0] - 1:
        neighbours.append([z+1, y, x])
    if y > 0:
        neighbours.append([z, y-1, x])
    if y < image.shape[1] - 1:
        neighbours.append([z, y+1, x])
    if x > 0:
        neighbours.append([z, y, x-1])
    if x < image.shape[2] - 1:
        neighbours.append([z, y, x+1])
    return neighbours

def new_neighbours(coords, image, processed_pixels):
    neighbours = []
    for [z, y, x] in possible_neighbours(coords, image):
        if image[z, y, x] != 0 and processed_pixels[z, y, x] == 0:
            processed_pixels[z, y, x] = 2
            neighbours.append([z, y, x])
    return neighbours

def remove_connected_components(image, threshold = 10000, verbose = True):
    connected_components = []
    processed_pixels = (image == 0).astype(int)

    while not np.all(processed_pixels):
        [z, y, x] = np.argwhere(processed_pixels == 0)[0]
        processed_pixels[z, y, x] = 1
        component = [[z, y, x]]
        queue = new_neighbours([z, y, x], image, processed_pixels)
        while(len(queue) != 0):
            [z, y, x] = queue.pop()
            processed_pixels[z, y, x] = 1
            component.append([z, y, x])
            queue.extend(new_neighbours([z, y, x], image, processed_pixels))
        connected_components.append(component)

    remaining_components = 0
    for component in connected_components:
        if len(component) < threshold:
            for [z, y, x] in component:
                image[z, y, x] = 0
        else:
            remaining_components += 1
    if verbose:
        print("Remaining components of volume >=", threshold, ":", remaining_components)


if __name__ == "__main__":
    assert(len(sys.argv) == 3)
    image_path = sys.argv[1]
    output_path = sys.argv[2]
    
    print("Reading image", image_path)
    image = sitk.GetArrayFromImage(sitk.ReadImage(image_path))

    remove_connected_components(image)
    sitk.WriteImage(sitk.GetImageFromArray(image), output_path)
    print("Saving result to", output_path)

