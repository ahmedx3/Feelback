import cv2
from utils import verbose
from utils import io


def main():
    args = io.get_command_line_args()
    input_video = args.input_video

    video = io.read_video(input_video)

    frame_number = 0
    # Read frame by frame until video is completed
    while video.isOpened():
        frame_number += 1

        ret, frame = video.read()
        if ret == True: # Pipeline is implemented Here
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            verbose.imshow(frame, delay=1)
            verbose.print(f"Processing Frame #{frame_number}")

            # ============================================= Preprocessing =============================================

            
            # ============================================= Face Detection ============================================

            
            # ========================================= Emotion Classification ========================================

            
            # ============================================= Face Tracking =============================================

            
            # ============================================ Gaze Estimation ============================================

            
            # ==================================== Profiling (Age/Gender Detection) ===================================

            
            # =========================================== Integrate Modules ===========================================

            
            # =============================================== Analytics ===============================================
            

        else: 
            break

    # When everything done, release the video capture object
    video.release()



if __name__ == '__main__':
    main()

