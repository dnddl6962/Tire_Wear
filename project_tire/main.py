import cv2
import numpy as np
import sys
from sklearn.cluster import KMeans
import matplotlib.image as mpimg
from matplotlib import pyplot as plt

# grayscale


def bgr2gray(bgr_img):
    # BGR 색상값
    b = bgr_img[:, :, 0]
    g = bgr_img[:, :, 1]
    r = bgr_img[:, :, 2]
    result = ((0.299 * r) + (0.587 * g) + (0.114 * b))
    # imshow 는 CV_8UC3 이나 CV_8UC1 형식을 위한 함수이므로 타입변환
    return result.astype(np.uint8)


input_bad = cv2.imread("bad.png", cv2.IMREAD_COLOR)  # 마모타이어 인풋
bgr_bad = bgr2gray(input_bad)
cv2.namedWindow('GrayScale bad')
# 지정한윈도우에 이미지를 보여준다.
cv2.imshow("GrayScale bad", bgr_bad)
# 지정한 시간만큼 사용자의 키보드입력을 대기한다. 0으로하면 키보드대기입력을 무한히 대기하도록한다.
cv2.waitKey(0)
cv2.imwrite("bgr2gray_bad.jpg", bgr_bad)

input_good = cv2.imread("good.png", cv2.IMREAD_COLOR)  # 정상타이어 인풋
bgr_good = bgr2gray(input_good)
cv2.namedWindow('GrayScale good')
# 지정한윈도우에 이미지를 보여준다.
cv2.imshow("GrayScale good", bgr_good)
# 지정한 시간만큼 사용자의 키보드입력을 대기한다. 0으로하면 키보드대기입력을 무한히 대기하도록한다.
cv2.waitKey(0)
cv2.imwrite("bgr2gray_good.jpg", bgr_good)

# bi
input_bgr_bad = cv2.imread("bgr2gray_bad.jpg")

input_bgr_bad = cv2.resize(input_bgr_bad, dsize=(0, 0), fx=2, fy=2)

dst1 = cv2.bilateralFilter(input_bgr_bad, -1, 10, 10)

cv2.imshow("dst1", dst1)


cv2.waitKey()
cv2.imwrite("bi_bad.jpg", dst1)


input_bgr_good = cv2.imread("bgr2gray_good.jpg")

input_bgr_good = cv2.resize(input_bgr_good, dsize=(0, 0), fx=2, fy=2)

dst2 = cv2.bilateralFilter(input_bgr_good, -1, 10, 10)

cv2.imshow("dst2", dst2)

cv2.waitKey()
cv2.imwrite("bi_good.jpg", dst2)


# 2진화

src1 = cv2.imread("bi_bad.jpg")
ret, res1 = cv2.threshold(src1, 100, 255, cv2.THRESH_BINARY)
cv2.imshow("2_bad", res1)
cv2.waitKey()
cv2.imwrite("thr_bad.jpg", res1)


src2 = cv2.imread("bi_good.jpg")
ret, res2 = cv2.threshold(src2, 100, 255, cv2.THRESH_BINARY)
cv2.imshow("2_good", res2)
cv2.waitKey()
cv2.imwrite("thr_good.jpg", res2)

cv2.destroyAllWindows()


res1 = cv2.cvtColor(res1, cv2.COLOR_BGR2RGB)
res2 = cv2.cvtColor(res2, cv2.COLOR_BGR2RGB)

print(res1.shape)
print(res2.shape)

res1 = res1.reshape((res1.shape[0] * res1.shape[1], 3))
res2 = res2.reshape((res2.shape[0] * res2.shape[1], 3))

print(res1.shape)
print(res2.shape)

k = 2  # 예제는 5개로 나누겠습니다
clt1 = KMeans(n_clusters=k)
clt1.fit(res1)

clt2 = KMeans(n_clusters=k)
clt2.fit(res2)


def centroid_histogram(clt):
    # grab the number of different clusters and create a histogram
    # based on the number of pixels assigned to each cluster
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)

    # normalize the histogram, such that it sums to one
    hist = hist.astype("float")
    hist /= hist.sum()

    # return the histogram
    return hist


hist1 = centroid_histogram(clt1)
hist2 = centroid_histogram(clt2)
print(hist1)
print(hist2)
