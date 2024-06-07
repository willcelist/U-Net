from u_net_model import multi_unet_model
import tensorflow as tf
import os
import glob
import cv2
import numpy as np
from matplotlib import pyplot as plt
from keras.utils.np_utils import normalize


def load_train_img(img_dir, SIZE_Y, SIZE_X):
    train_img_arr = []
    for img_folder in os.listdir(img_dir):
        img_path = str(os.path.join(img_dir, img_folder, 'img.png'))
        img = cv2.imread(img_path, 0)
        img = cv2.resize(img, (SIZE_Y, SIZE_X), interpolation=cv2.INTER_NEAREST)
        train_img_arr.append(img)
    train_img_arr = np.array(train_img_arr)
    print('number of images', len(train_img_arr))
    return train_img_arr


def load_train_mask(dir_masks, SIZE_Y, SIZE_X):
    train_mks = []
    for mask_folder in os.listdir(dir_masks):
        mask_path = os.path.join(dir_masks, mask_folder, 'label.png')
        mask = cv2.imread(mask_path, 0)
        mask = cv2.resize(mask, (SIZE_Y, SIZE_X), interpolation=cv2.INTER_NEAREST)
        train_mks.append(mask)
    classes = len(np.unique(train_mks))
    train_mks = np.array(train_mks)
    print('number of masks', len(train_mks))
    print('number of classes = ', np.unique(train_mks))
    return train_mks, classes


def encoder(masks_train, images_train):
    # encode labels, flatten, encode and reshape multi dim array
    from sklearn.preprocessing import LabelEncoder
    labelencoder = LabelEncoder()
    n, h, w = masks_train.shape
    train_masks_shaped_1 = masks_train.reshape(-1, 1)
    train_masks_encode_1 = labelencoder.fit_transform(train_masks_shaped_1)
    train_masks_encoded_prime = train_masks_encode_1.reshape(n, h, w)

    np.unique(train_masks_encoded_prime)

    images_train = np.expand_dims(images_train, axis=3)
    images_train = normalize(images_train, axis=1)

    train_masks_in = np.expand_dims(train_masks_encoded_prime, axis=3)

    return images_train, train_masks_in, train_masks_encoded_prime, train_masks_encode_1


def split_10_90(images_arr_arg, masks_input_arg):
    # quick testing subset
    # 10% testing 90% training
    from sklearn.model_selection import train_test_split
    X1_1, X_test_1, y1_1, y_test_1, = train_test_split(images_arr_arg, masks_input_arg, test_size=0.10, random_state=0)
    # split training data for model quick testing
    X_train_1, x_1, y_train_1, y_1 = train_test_split(X1_1, y1_1, test_size=0.2, random_state=0)
    print('class values in data are: ', np.unique(X_train_1))
    return X1_1, X_test_1, y1_1, y_test_1, X_train_1, y_train_1


def categories(y_train_arg, y_test_arg, n_classes_arg, mask_encoded):
    from keras.utils.np_utils import to_categorical
    train_mask_cat_1 = to_categorical(y_train_arg, num_classes=n_classes_arg)
    y_train_cat_1 = train_mask_cat_1.reshape((y_train_arg.shape[0], y_train_arg.shape[1], y_train_arg.shape[2], n_classes_arg))

    test_mask_cat_1 = to_categorical(y_test_arg, num_classes=n_classes_arg)
    y_test_cat_1 = test_mask_cat_1.reshape((y_test_arg.shape[0], y_test_arg.shape[1], y_test_arg.shape[2], n_classes_arg))

    from sklearn.utils import class_weight
    class_weights_1 = class_weight.compute_class_weight(class_weight='balanced',
                                                      classes=np.unique(mask_encoded),
                                                      y=mask_encoded)
    class_weights_dict_1 = {i: class_weights_1[i] for i in range(len(class_weights_1))}
    print('class weights are: ', class_weights_dict_1)

    IMG_HEIGHT_1 = X_train.shape[1]
    IMG_WIDTH_1 = X_train.shape[2]
    IMG_CHANNELS_1 = X_train.shape[3]

    return IMG_CHANNELS_1, IMG_WIDTH_1, IMG_HEIGHT_1, class_weights_dict_1, class_weights_1, y_test_cat_1, test_mask_cat_1, y_train_cat_1, train_mask_cat_1




def get_model():
    return multi_unet_model(n_classes, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)



if __name__ == '__main__':
    SIZE_X = 128
    SIZE_Y = 128

    img_dir = r'\python_projects\labeled_images'

    images = load_train_img(img_dir, SIZE_Y, SIZE_X)
    masks, n_classes = load_train_mask(img_dir, SIZE_Y, SIZE_X)
    train_images, train_masks_input, train_masks_encoded_original, train_masks_encode = encoder(masks, images)
    X1, X_test, y1, y_test, X_train, y_train = split_10_90(train_images, train_masks_input)

    IMG_CHANNELS, IMG_WIDTH, IMG_HEIGHT, class_weights_dict, class_weights, y_test_cat, test_mask_cat, y_train_cat, train_mask_cat = categories(y_train, y_test, n_classes, train_masks_encode)

    model = get_model()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    # if starting with pretrained weights
    # model.load_weights('???.hdf5')

    history = model.fit(X_train, y_train_cat, batch_size=16, verbose="auto", epochs=50,
                        validation_data=(X_test, y_test_cat), shuffle=False)

    model.save('sandstone_50epoch.hdf5')

    _, acc = model.evaluate(X_test, y_test_cat)
    print('accuracy is = ', (acc * 100), '%')

