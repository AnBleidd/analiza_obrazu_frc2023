import cv2
import numpy as np

#zamienianie obrazu na czarno-biały
img = cv2.imread('apriltagi.png', cv2.IMREAD_GRAYSCALE)
img_color = cv2.imread('apriltagi.png')

#skalowanie obrazu
scale_percent = 60 # percent of original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)

img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
img_color = cv2.resize(img_color, dim, interpolation = cv2.INTER_AREA)
print(img.shape)

from pupil_apriltags import Detector

at_detector = Detector(
   families="tag16h5",
   nthreads=1,
   quad_decimate=1.0,
   quad_sigma=0.0,
   refine_edges=1,
   decode_sharpening=0.25,
   debug=0
)
# Konwersja na numpy.ndarray
img_np = np.array(img)

# Konwersja na numpy.uint8
img_np = img_np.astype(np.uint8)
detection = at_detector.detect(img_np)
print(detection)

for tag in detection:
        for idx in range(len(tag.corners)):
            cv2.line(
               img_color ,
                tuple(tag.corners[idx - 1, :].astype(int)),
                tuple(tag.corners[idx, :].astype(int)),
                (0, 255, 0),
            )

        cv2.putText(
            img_color,
            str(tag.tag_id),
            org=(
                tag.corners[0, 0].astype(int) + 10,
                tag.corners[0, 1].astype(int) + 10,
            ),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.8,
            color=(0, 0, 255),
        )


cv2.imshow("Detected tags", img_color)

k = cv2.waitKey(0)
if k == 27:  # wait for ESC key to exit
    cv2.destroyAllWindows()


cv2.imshow('obraz', img)
cv2.waitKey(0)