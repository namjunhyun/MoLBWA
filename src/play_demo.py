import cv2, os, time
HERE=os.path.dirname(os.path.abspath(__file__))
path=os.path.join(HERE,"..","data","eye_tracking_demo.mp4")
win="Eye Tracking Demo (q=quit)"
cv2.namedWindow(win, cv2.WINDOW_NORMAL)
cv2.resizeWindow(win, 960, 720)
while True:
    cap=cv2.VideoCapture(path)
    while True:
        ret,f=cap.read()
        if not ret: break
        cv2.imshow(win,f)
        if (cv2.waitKey(33)&0xFF)==ord('q'):
            cap.release(); cv2.destroyAllWindows(); raise SystemExit
    cap.release()
