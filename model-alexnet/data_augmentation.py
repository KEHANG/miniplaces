import numpy as np
import scipy.misc
import imgaug as ia
from imgaug import augmenters as iaa

ia.seed(1)

### Define augmentation sequence ###
class Augmentator():
    def __init__(self):
        self.seq = iaa.Sequential([
        iaa.Fliplr(0.5), # horizontal flips
        # crops mess up image due to black edges
        iaa.CropAndPad(
            percent=(0, 0.1),
            pad_mode=["edge"]), # random crops
        # Small gaussian blur with random sigma between 0 and 0.5.
        # But we only blur about 50% of all images.
        iaa.Sometimes(0.5,iaa.GaussianBlur(sigma=(0, 0.05))),
        # Strengthen or weaken the contrast in each image.
        iaa.ContrastNormalization((0.75, 1.5)),
        # Add gaussian noise.
        # For 50% of all images, we sample the noise once per pixel.
        # For the other 50% of all images, we sample the noise per pixel AND
        # channel. This can change the color (not only brightness) of the
        # pixels.
        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.01*255), per_channel=0.5),
        # Make some images brighter and some darker.
        # In 20% of all cases, we sample the multiplier once per channel,
        # which can end up changing the color of the images.
        iaa.Multiply((0.8, 1.2), per_channel=0.2),
        # Apply affine transformations to each image.
        # Scale/zoom them, translate/move them, rotate them and shear them.
        iaa.Affine(
            scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
            translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
            rotate=(-12.5, 12.5),
            shear=(-4, 4)
            )
        ], random_order=True) # apply augmenters in random order
