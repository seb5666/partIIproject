import numpy as np

def normalize_scans(images, num_channels):
        normalized_images = []
        for image in images:
                normalized_image = np.empty_like(image)
                for channel in range(num_channels):
                    normalized_image[:,:,:, channel] = normalize_channel(image[:,:,:, channel])
                normalized_images.append(normalized_image)
        return normalized_images

def normalize_channel(channel):
    std = np.std(channel)
    if std != 0:
        return (channel - np.mean(channel)) / std
    else:
        return channel
