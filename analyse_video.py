from imutils.video import VideoStream
from imutils.video import FPS
import analyse_image_file
from analyse_image_file import *




def video_detector(from_webcam):

    if not from_webcam:
        fps_count = int(vs.get(cv2.CAP_PROP_FRAME_COUNT))
        print('FPS count:', fps_count)

    analyse_image = Analyse_image()

    while True:

        if from_webcam:
            frame = vs.read()
            analyse_image.from_frame(frame)
        
        else:
            frame_position_count = int(vs.get(cv2.CAP_PROP_POS_FRAMES))
            # print('Frame position:', frame_position_count)

            ret, frame = vs.read()
        
            frame_width, frame_height = (
                int(vs.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT))
            )

            if not ret:
                break

            if frame_position_count % 10 == 0:
                try:
                    
                    analyse_image.from_frame(frame)
                
                except Exception:
                    pass
        
            
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

        fps.update()

    fps.stop()
    print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

    if from_webcam:
        vs.stop()
    else:
        vs.release()

    cv2.destroyAllWindows()











if __name__ == "__main__":

    video_input = str(input("Enter relative path of the video file or press enter to use webcam: "))
    if video_input == "":
        vs = VideoStream(src=0).start()
        from_webcam = True
    else:
        vs = VideoCapture(video_input)
        from_webcam = False

    print("[INFO] starting video stream...")
    time.sleep(1.0)
    fps = FPS().start()

    video_detector(from_webcam)
    # analyse_image = Analyse_image()


