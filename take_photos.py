import cv2
from GUI import start_GUI

def take_photos():
    cap = cv2.VideoCapture(0)
    num = 1

    while cv2.waitKey(1) == -1:
        # get a frame
        ret, frame = cap.read()
        # show a frame
        cv2.imshow("capture", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.imwrite("pictures/zyc/p_" + str(num) + ".jpg", frame)
            print("Please wait")
            print("Photo " + str(num) + " has been taken, please continue")
            num += 1
        if cv2.waitKey(1) & 0xFF == ord('w'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    #start_GUI()
    take_photos()