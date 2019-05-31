import cv2

def convert2Binary(img_grayscale):
    img_binary = cv2.adaptiveThreshold(img_grayscale,
                                        maxValue=255,
                                        adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
                                        thresholdType=cv2.THRESH_BINARY,
                                        blockSize=15,
                                        C=8)
    return img_binary

def main():
    video = cv2.VideoCapture(0)
    while(True):
        ret, frame = video.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        binary = convert2Binary(gray)
        cv2.imshow("", binary)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()