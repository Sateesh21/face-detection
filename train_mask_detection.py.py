#import the Necessary libraries
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os


"""Initialize the 
		initial learning rate,
		number of epochs,
		and batch size"""

INIT_LR = 0.001
EPOCHS = 100
BS = 32

#Creating two empty lists
data =[]    #Used to append the Image Arrays after looping
labels =[]  #used to Intialize the Corresponding images either With or WithOut Mask Images


#Looping the DATASETS
for category in CATEGORIES:                  
    path = os.path.join(DIRECTORY, category)
    for img in os.listdir(path):                           #Looping Each and every Images present in both folders
    	img_path = os.path.join(path, img)
    	image = load_img(img_path, target_size=(224, 224)) #target_size is for Height and weight of a image
    	image = img_to_array(image) 
    	image = preprocess_input(image)

    	data.append(image)
    	labels.append(category) #appending

        
""" "encoding on the labels",
	 to convert from Alphabetical values to Numeric Values"""
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

#From Numerical values to Arrays
data = np.array(data, dtype="float32")
labels = np.array(labels)

#Splitting data arrays into two subsets for Training and Testing
(trainX, testX, trainY, testY) = train_test_split(data, labels,
	test_size=0.20, stratify=labels, random_state=42)

# Using ImageDataGenerator to Generate number of images by using single Image

aug = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest")


# load the MobileNetV2 network 
baseModel = MobileNetV2(weights="imagenet", include_top=False,
                        input_tensor=Input(shape=(224, 224, 3)))

#base model
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)


#Model Function and this the actual model will train
model = Model(inputs=baseModel.input, outputs=headModel)

#Looping over all layers in the base model and Freeze
for layer in baseModel.layers:
    layer.trainable = False
    
#Compilation our module
print("Loading compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,
              metrics=["accuracy"])


# train the head of the network
print("training head...")
H = model.fit(
    aug.flow(trainX, trainY, batch_size=BS),
    steps_per_epoch=len(trainX) // BS,
    validation_data=(testX, testY),
    validation_steps=len(testX) // BS,
    epochs=EPOCHS)


# Predictions on the testing set
print("[INFO] evaluating network...")
predIdxs = model.predict(testX, batch_size=BS)

# Indexing
predIdxs = np.argmax(predIdxs, axis=1)

# show a nicely formatted classification report
print(classification_report(testY.argmax(axis=1), predIdxs,
                            target_names=lb.classes_))

# serialize the model to disk
print("saving mask detector model...")
model.save("mask_detector.model", save_format="h5")

# Plotting by using MatPlotLib
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.png")

