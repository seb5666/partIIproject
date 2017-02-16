import sys
import numpy as np
import SimpleITK as sitk
assert(len(sys.argv) == 3)
image_path = sys.argv[1]
output_path = sys.argv[2]

print("Reading image", image_path)

image = sitk.GetArrayFromImage(sitk.ReadImage(image_path))

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


def new_neighbours(coords, image, pixels_to_process, queue= []):
    neighbours = []
    for [z, y, x] in possible_neighbours(coords, image):
        if image[z, y, x] != 0 and [z,y,x] not in queue and [z, y, x] in pixels_to_process:
            neighbours.append([z, y, x])
    return neighbours

def remove_connected_components(image, threshold = 10000):
    connected_components = []

    #pixels_to_process = [(z, y, x) for z in range(image.shape[0]) for y in range(image.shape[1]) for x in range(image.shape[2])

    pixels_to_process = np.argwhere(image != 0).tolist()
    while(len(pixels_to_process) != 0):
        print(len(pixels_to_process))
        [z, y, x] = pixels_to_process.pop()
        if image[z, y, x] != 0:
            component = [[z, y, x]]
            queue = new_neighbours([z, y, x], image, pixels_to_process)
            while(len(queue) != 0):
                voxel_coords = queue.pop()
                if voxel_coords in pixels_to_process:
                    pixels_to_process.remove(voxel_coords)
                    component.append(voxel_coords)
                    queue.extend(new_neighbours(voxel_coords, image, pixels_to_process, queue))

            connected_components.append(component) 
         
    for component in connected_components:
        print(len(component))
        print(component)
        if len(component) < threshold:
            for [z, y, x] in component:
                image[z, y, x] = 0

def new_neighbours2(coords, image, processed_pixels):
    neighbours = []
    for [z, y, x] in possible_neighbours(coords, image):
        if image[z, y, x] != 0 and processed_pixels[z, y, x] == 0:
            processed_pixels[z, y, x] = 2
            neighbours.append([z, y, x])
    return neighbours

def remove_connected_components2(image, threshold = 10000):
    connected_components = []
    processed_pixels = (image == 0).astype(int)

    while not np.all(processed_pixels):
        [z, y, x] = np.argwhere(processed_pixels == 0)[0]
        print([z,y,x])
        processed_pixels[z, y, x] = 1
        component = [[z, y, x]]
        queue = new_neighbours2([z, y, x], image, processed_pixels)
        while(len(queue) != 0):
            [z, y, x] = queue.pop()
            processed_pixels[z, y, x] = 1
            component.append([z, y, x])
            queue.extend(new_neighbours2([z, y, x], image, processed_pixels))
        connected_components.append(component)

    for component in connected_components:
        print(len(component))
        if len(component) < threshold:
            for [z, y, x] in component:
                image[z, y, x] = 0



#test = np.array([[[1,1,1],[0,0,0],[0,0,0]],[[0,0,0],[1,1,1],[0,0,0]],[[0,0,0],[0,1,0],[0,0,0]]])
#print(test)
#remove_connected_components2(test)
#print(test)
remove_connected_components2(image)
sitk.WriteImage(sitk.GetImageFromArray(image), output_path)
print("Saving result to", output_path)

