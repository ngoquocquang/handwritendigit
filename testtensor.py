import tensorflow as tf
import cv2
from modeltensor import LeNet_5
import matplotlib.pyplot as plt
import numpy as np

def main(img_path):
    model_path = "model/Model_Lenet-5.ckpt"


    x = tf.placeholder(tf.float32, (None,)+(28,28,1))
    y = tf.placeholder(tf.float32, (None, 10))

    lenet5 = LeNet_5(x,y)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, model_path)
        img = cv2.imread(img_path)
        grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        cvt_img = np.zeros((1,28,28,1))

        for i in range(28):
            for j in range(28):
                cvt_img[0][i][j][0] = grey_img[i][j]

        pred = sess.run(lenet5.pred, {x: cvt_img, lenet5.keep_prob: 1.0})[0]
        plt.title("Number Predict: " + str(pred))
        plt.imshow(img)
        plt.show()

if __name__ == '__main__':
    main(img_path)