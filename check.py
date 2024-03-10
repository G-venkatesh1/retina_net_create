import cv2
import numpy as np


def drawBox(boxes, image):
    for i in range(0, len(boxes)):
        # changed color and width to make it visible
        cv2.rectangle(image, (boxes[i][2], boxes[i][3]), (boxes[i][4], boxes[i][5]), (255, 0, 0), 1)
    cv2.imshow("img", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def cvTest():
    # imageToPredict = cv2.imread("img.jpg", 3)
    imageToPredict = cv2.imread("samp.png", 3)
    print(imageToPredict.shape)

    # Note: flipped comparing to your original code!
    # x_ = imageToPredict.shape[0]
    # y_ = imageToPredict.shape[1]
    y_ = imageToPredict.shape[0]
    x_ = imageToPredict.shape[1]

    targetSize = 416
    x_scale = targetSize / x_
    y_scale = targetSize / y_
    print(x_scale, y_scale)
    img = cv2.resize(imageToPredict, (targetSize, targetSize));
    print(img.shape)
    img = np.array(img);

    # original frame as named values
    (origLeft, origTop, origRight, origBottom) = (160, 35, 555, 470)

    x = int(np.round(origLeft * x_scale))
    y = int(np.round(origTop * y_scale))
    xmax = int(np.round(origRight * x_scale))
    ymax = int(np.round(origBottom * y_scale))
    # Box.drawBox([[1, 0, x, y, xmax, ymax]], img)
    drawBox([[1, 0, x, y, xmax, ymax]], img)


cvTest()