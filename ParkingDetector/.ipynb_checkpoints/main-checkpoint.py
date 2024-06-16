import cv2

from util import get_parking_spots_bboxes

mask = 'mask_crop.png'
video_path = 'parking_crop_loop.mp4'


mask = cv2.imread(mask, 0)
cap = cv2.VideoCapture(video_path)

connected_components = cv2.connectedComponents(mask, 4, cv2.CV2_32S)

get_parking_spots_bboxes(connected_components)
ret = True
while ret:
    ret, frame = cap.read()

    cv2.imshow('frame', frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()