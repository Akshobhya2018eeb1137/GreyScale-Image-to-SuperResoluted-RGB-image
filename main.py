import os
import re
from scipy import ndimage, misc
from skimage.transform import resize, rescale
from matplotlib import pyplot
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(0)

from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.layers import Conv2DTranspose, UpSampling2D, add
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
import tensorflow as tf
print(tf.__version__)

input_img = Input(shape = (32, 32, 1))
l1 = Conv2D(64, (3, 3), padding = 'same', activation = 'relu', activity_regularizer = regularizers.l1(1e-9))(input_img)
l2 = Conv2D(64, (3, 3), padding = 'same', activation = 'relu', activity_regularizer = regularizers.l1(1e-9))(l1)
#l3 = Conv2D(128, (3, 3), padding = 'same', activation = 'relu', activity_regularizer = regularizers.l1(1e-9))(l2)
l3 = MaxPooling2D(padding = 'same')(l2)
l4 = Conv2D(128, (3, 3), padding = 'same', activation = 'relu', activity_regularizer = regularizers.l1(1e-9))(l3)
l5 = Conv2D(128, (3, 3), padding = 'same', activation = 'relu', activity_regularizer = regularizers.l1(1e-9))(l4)
l6 = MaxPooling2D(padding = 'same')(l5)
l7 = Conv2D(256, (3, 3), padding = 'same', activation = 'relu', activity_regularizer = regularizers.l1(1e-9))(l6)
encoder = Model(input_img, l7)

#replace maxpooling with upsampling and apply residual connection from all layers above maxpooling layer to 2 layers after
#upsampling like here from l5 to l10 and also l14 to l2
l8 = UpSampling2D()(l7)
l9 = Conv2D(128, (3, 3), padding = 'same', activation = 'relu', activity_regularizer = regularizers.l1(1e-9))(l8)
l10 = Conv2D(128, (3, 3), padding = 'same', activation = 'relu', activity_regularizer = regularizers.l1(1e-9))(l9)
l11 = add([l5, l10])
l12 = UpSampling2D()(l11)
l13 = Conv2D(64, (3, 3), padding = 'same', activation = 'relu', activity_regularizer = regularizers.l1(1e-9))(l12)
l14 = Conv2D(64, (3, 3), padding = 'same', activation = 'relu', activity_regularizer = regularizers.l1(1e-9))(l13)
l15 = add([l14, l2])
#below 3 filters because we want 3 channels RGB
decoded_output =  Conv2D(3, (3, 3), padding = 'same', activation = 'relu', activity_regularizer = regularizers.l1(1e-9))(l15)
autoencoder = Model(input_img, decoded_output)
autoencoder.summary()

autoencoder.compile(optimizer = 'adadelta', loss = 'mean_squared_error')

from google_images_download import google_images_download as down
response = down.googleimagesdownload()
arguments = {"keywords":"artwork, painting, old arts", "limit":100}
paths = response.download(arguments)

from  keras.preprocessing.image import ImageDataGenerator
path = r'C:\Users\Akshobhya\Downloads\artwork'
train_data = ImageDataGenerator(rescale = 1./255)
train = train_data.flow_from_directory(path, target_size = (32, 32), batch_size = 910, class_mode = None)

from PIL import Image
import glob

image_list = []
resized_images = []

for filename in glob.glob('C:\\Users\\Akshobhya\\Downloads\\jho\\*.jpg'):
    print(filename)
    img = Image.open(filename)
    image_list.append(img)

for image in image_list:
    image = image.resize((32, 32))
    resized_images.append(image)

for (i, new) in enumerate(resized_images):
    new.save('{}{}{}'.format('C:\\Users\\Akshobhya\\Downloads\\final\\', i+1, '.jpg'))
    
from skimage import io

from PIL import Image
import os
from imutils import paths

# grab the list of images in our dataset directory
paths = list(paths.list_images(r'C:\Users\Akshobhya\Downloads\final'))

color = []
grey = []

def makeDataset():
    cnt = 0
    # loop over the image paths
    for item in paths:
        print(item)
        cnt += 1
        img = io.imread(item, as_gray=False)
        img2 = io.imread(item, as_gray=True)
        color.append(img)
        grey.append(img2)
            # load the image and resize it to 224 x 224 Pixels
           # im = Image.open(item)
            #imResize = im.resize((224,224), Image.ANTIALIAS)
            # save image as JPEG
            #imResize.save(item, 'JPEG', quality=90)
#img = io.imread('image.png', as_gray=True)
makeDataset()
print(len(color))
print(len(grey))
color = np.array(color)
grey = np.array(grey)
color = color / 255.
grey = grey / 255.
plt.imshow(grey[0], cmap = 'gray' ) 
plt.imshow(color[0])

autoencoder.fit(grey, color, epochs = 300, batch_size = 20, shuffle = True, validation_split = 0.15)
x = autoencoder.predict(grey)
plt.imshow(x[0])
