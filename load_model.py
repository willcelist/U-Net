import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from main import load_train_img
from main import load_train_mask
from main import encoder
from main import split_10_90

SIZE_X = 128
SIZE_Y = 128

img_dir = r'C:\Classification\python_projects\labeled_images'

images = load_train_img(img_dir, SIZE_Y, SIZE_X)
masks, n_classes = load_train_mask(img_dir, SIZE_Y, SIZE_X)
train_images, train_masks_input, train_masks_encoded_original, train_masks_encode = encoder(masks, images)
X1, X_test, y1, y_test, X_train, y_train = split_10_90(train_images, train_masks_input)


loaded_model = tf.keras.models.load_model('sandstone_50epoch_catXentropy.hdf5')
loaded_model.summary()

tes_img_number = random.randint(0, len(X_test)-1)
test_img = X_test[tes_img_number]
ground_truth = y_test[tes_img_number]
test_img_norm = test_img[:,:,0][:,:,None]
test_img_input = np.expand_dims(test_img_norm, 0)
prediction = (loaded_model.predict(test_img_input))
predicted_img = np.argmax(prediction, axis=3)[0, :, :]
print('prediction: ', predicted_img)

plt.figure(figsize=(12, 8))
plt.subplot(231)
plt.title('Testing Image')
plt.imshow(test_img[:, :, 0], cmap='gray')
plt.subplot(232)
plt.title('Testing Label')
plt.imshow(ground_truth[:, :, 0], cmap='jet')
plt.subplot(233)
plt.title('Prediction on test image')
plt.imshow(predicted_img, cmap='jet')
plt.show()


