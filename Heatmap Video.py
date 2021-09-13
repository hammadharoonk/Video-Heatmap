import numpy as np
import cv2
import copy


def change_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.add(v, value)
    v[v > 255] = 255
    v[v < 0] = 0
    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

def rolling(image_list, history):
    img = image_list[0]
    new = np.zeros(img.shape, img.dtype)
    for i in range(history):
        new = cv2.addWeighted(image_list[i],1, new,1,0.0)
    return new


def main():
    cap = cv2.VideoCapture(r"roundabout_03_roi.mp4")
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"fps: {fps}")
    # pip install opencv-contrib-python
    fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))//2
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))//2

    alpha = float(1.0)
    beta = int(0)
    out = cv2.VideoWriter(r"C:\Users\KHAN\Desktop\computervision\motiondetection\heatmap\roundabout_04.mp4", fourcc, fps*2, (width, height), True)

    # number of frames is a variable for development purposes, you can change the for loop to a while(cap.isOpened()) instead to go through the whole video
    # num_frames = 1600
    # start_frames = 6000
    fcount = 0
    first_iteration_indicator = 1
    fifo = []
    history = 100
    for i in range(0, length-10):
        fcount += 1
        print(f"{fcount}/{length} frames")
        if fcount > 300:
            history = 300

        if (first_iteration_indicator == 1):
            ret, frame = cap.read()
            frame = cv2.resize(frame, (width, height), interpolation = cv2.INTER_AREA)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            height, width = gray.shape[:2]
            accum_image = np.zeros((height, width), np.uint8)
            final_image = np.zeros((height, width), np.uint8)
            blank = np.zeros((height, width), np.uint8)
            first_iteration_indicator = 0
            new_image = np.zeros(frame.shape, frame.dtype)

        else:
            # read a frame
            ret, frame = cap.read()
            if ret is None:
                break


            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # convert to grayscale

            fgmask = fgbg.apply(gray)  # remove the background

            # for testing purposes, show the result of the background subtraction
            # cv2.imshow('diff-bkgnd-frame', fgmask)

            # apply a binary threshold only keeping pixels above thresh and setting the result to maxValue.  If you want
            # motion to be picked up more, increase the value of maxValue.  To pick up the least amount of motion over time, set maxValue = 1
            thresh = 2
            maxValue = 5
            ret, th1 = cv2.threshold(fgmask, thresh, maxValue, cv2.THRESH_BINARY)
            # cv2.imshow("threshold", th1)
            final_image = cv2.addWeighted(th1, 1, final_image, 1, 0)
            if len(fifo)<history:
                fifo.append(th1)
                accum_image = cv2.addWeighted(th1, 1, accum_image, 1, 0)
            else:
                fifo.pop(0)
                fifo.append(th1)
                accum_image = rolling(fifo, history)

            color_image = cv2.applyColorMap(accum_image, cv2.COLORMAP_OCEAN)
            color_image = cv2.GaussianBlur(color_image, (7,7), cv2.BORDER_WRAP)

            cv2.imshow("color final",color_final)

            cv2.imshow("frame", frame)
            out.write(frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
              break

    cv2.imwrite(r"final_color.jpg",color_final)
    # cleanup
    cap.release()
    out.release()
    out2.release()
    out3.release()
    cv2.destroyAllWindows()

if __name__=='__main__':
    main()