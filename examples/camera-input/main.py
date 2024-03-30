import cv2
from models.mtcnn import MTCNN

video_input = cv2.VideoCapture(0)
model = MTCNN()

while True:
    ret, frame = video_input.read()

    if not ret:
        break

    boxes, _ = model.detect(frame)

    if boxes is None:
        continue

    for box in boxes:
        x, y, w, h = box
        x, y, w, h = int(x), int(y), int(w), int(h)

        cv2.rectangle(frame, (x, y), (w, h), (255, 0, 0), 2)

    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_input.release()
cv2.destroyAllWindows()
