gitimport cv2

def main():
    video = cv2.VideoCapture(0)
    while(True):
        ret, frame = video.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        new_frame = cv2.equalizeHist(frame)

        cv2.imshow("", frame)
        cv2.imshow("a", new_frame)
        if (cv2.waitKey(1) & 0xFF == ord('q')):
            break
    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()