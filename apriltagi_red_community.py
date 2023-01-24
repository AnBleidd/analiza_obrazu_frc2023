import cv2 as cv
import numpy as np
from pupil_apriltags import Detector

capture = cv.VideoCapture('img/apriltags_1.mp4')

capture = cv.VideoCapture('img/apriltags_1.mp4', cv.IMREAD_GRAYSCALE)
capture_color = cv.VideoCapture('img/apriltags_1.mp4')

capture_np = np.array(capture)
capture_np = capture_np.astype(np.uint8)

while True:
    isTrue, frame = capture_color.read()
    cv.imshow('apriltagi', capture_color)
    if cv.waitKey(7) and  0xFF == ord('d'):
        break
capture.release()
cv.destroyAllWindows()

scale_percent = 60 # percent of original size
width = int(frame.shape[1] * scale_percent / 100)
height = int(frame.shape[0] * scale_percent / 100)
dim = (width, height)


at_detector = Detector(
   families="tag16h5",
   nthreads=1,
   quad_decimate=1.0,
   quad_sigma=0.0,
   refine_edges=1,
   decode_sharpening=0.25,
   debug=0
)

frame_np = np.array(frame)
frame_np = frame_np.astype(np.uint8)
detection = at_detector.detect(frame_np)
print(detection)

for tag in detection:
        for idx in range(len(tag.corners)):
            cv.line(
               capture_color ,
                tuple(tag.corners[idx - 1, :].astype(int)),
                tuple(tag.corners[idx, :].astype(int)),
                (0, 255, 0),
            )

        cv.putText(
            capture_color,
            str(tag.tag_id),
            org=(
                tag.corners[0, 0].astype(int) + 10,
                tag.corners[0, 1].astype(int) + 10,
            ),
            fontFace=cv.FONT_HERSHEY_SIMPLEX,
            fontScale=0.8,
            color=(0, 0, 255),
        )

cv.imshow("Detected tags", capture_color)

k = cv.waitKey(0)
if k == 27:  # wait for ESC key to exit
    cv.destroyAllWindows()


cv.imshow('obraz', frame)
cv.waitKey(0)