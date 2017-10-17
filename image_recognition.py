import numpy as np
import pandas as pd
from MultiSvm import MultiSVM
from KMeans import KMeans

W = 6
K = 800


def load_data():
    data = pd.read_csv('Xtr.csv', header=None)
    data.drop(3072, axis=1, inplace=True)
    labels = pd.read_csv("Ytr.csv")
    y_train = labels.ix[:, 1].as_matrix()
    X_train = data.as_matrix()

    data = pd.read_csv('Xte.csv', header=None)
    data.drop(3072, axis=1, inplace=True)
    X_test = data.as_matrix()

    images_train = []
    for i in range(X_train.shape[0]):
        images_train.append(X_train[i, :].reshape(3, 32, 32).transpose(1, 2, 0))
    images_train = np.array(images_train)

    images_test = []
    for i in range(X_test.shape[0]):
        images_test.append(X_test[i, :].reshape(3, 32, 32).transpose(1, 2, 0))
    images_test = np.array(images_test)

    return images_train, y_train, images_test


def image_representation(images, w, clf):
    X_image_repres = []

    for img in images:
        img_repres = np.zeros([(32 - w + 1), (32 - w + 1), clf.k])
        for i in np.arange(0, 32 - w):
            for j in np.arange(32 - w):
                patch = img[i:i + w, j:j + w, :]
                k = clf.predict(patch.reshape(1, w * w * 3))
                k_vector = np.zeros(clf.k)
                k_vector[k[0]] = 1
                img_repres[i, j, :] = k_vector

        quad1 = np.sum(img_repres[:13, :13, :], axis=(0, 1))
        quad2 = np.sum(img_repres[:13, 13:, :], axis=(0, 1))
        quad3 = np.sum(img_repres[13:, :13, :], axis=(0, 1))
        quad4 = np.sum(img_repres[13:, 13:, :], axis=(0, 1))

        final_vector = np.concatenate((quad1, quad2, quad3, quad4))
        X_image_repres.append(final_vector)

    return np.array(X_image_repres)


def bag_of_visual_words(images_train):
    patches = []
    for i in np.arange(int(images_train.shape[0])):
        for _ in np.arange(10):
            a = np.random.randint(0, 32 - W)
            b = np.random.randint(0, 32 - W)
            patch = images_train[i][a:a + W, b:b + W, :]
            patches.append(patch)
    patches = np.array(patches)

    k_means_train = patches.reshape(patches.shape[0], W * W * 3)

    return k_means_train


def make_submission(y_predict, file_name):
    file_obj = open(file_name, 'w')
    string = "Id,Prediction\n"
    file_obj.write(string)

    i = 1
    for digit in y_predict:
        string = str(i) + ',' + str(digit) + '\n'
        file_obj.write(string)
        i += 1

    file_obj.close()


def main():
    images_train, y_train, images_test = load_data()

    k_means_train = bag_of_visual_words(images_train)
    clf = KMeans(K)
    clf.train(k_means_train)

    X_train_image_repres = image_representation(images_train, W, clf)
    X_test_image_repres = image_representation(images_test, W, clf)

    my_svm = MultiSVM(n_classes=10)
    my_svm.train(X_train_image_repres, y_train)

    predictions = my_svm.predict(X_test_image_repres)

    make_submission(predictions, 'Yte.csv')


if __name__ == '__main__':
    main()