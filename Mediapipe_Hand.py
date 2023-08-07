import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os.path as osp
import imutils

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


def extract_skeleton_from_webcam(action='ok',label=7):
    Data_path = 'Sketeton_Data/ok'

    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands
    cap = cv2.VideoCapture(0)
    video_num = 0
    with mp_hands.Hands(
            model_complexity=1,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands:
        video = []
        while True:

            success, image = cap.read()
            h, w, _ = image.shape
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image)

            # Draw the hand annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            skeleton = []
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:

                    hand_list = list(hand_landmarks.landmark)
                    instance = []
                    for keypoint in hand_list:
                        instance.append([keypoint.x*w, keypoint.y*h, keypoint.z])

                    skeleton.append(np.array(instance).flatten())

                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
                num_hand = len(results.multi_hand_landmarks)
                if num_hand == 1:
                    skeleton.append(np.zeros(21*3))
                skeleton_arr = np.array(skeleton).flatten()
            else:
                skeleton_arr = np.zeros(21*2*3)
            video.append(skeleton_arr)

            # Flip the image horizontally for a selfie-view display.
            image_flip = cv2.flip(image, 1)


            image_flip = cv2.putText(image_flip, 'class {} video {}'.format(action, video_num),
                                         (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            image_flip = imutils.resize(image_flip, width=800)
            cv2.imshow('MediaPipe Hands', image_flip)

            if cv2.waitKey(1) & 0xFF == 32:
                print(f'save video {video_num} {action}')
                video_data = np.array(video)
                print(video_data.shape)

                anno = {'keypoints': video_data, 'label': label, 'class_name': action}
                path2save = osp.join(Data_path,'video{}_{}.pkl'.format(video_num, action))
                pd.to_pickle(anno, path2save)
                video_num += 1
                video = []
                while True:
                    if cv2.waitKey(1) & 0xFF == 32:
                        break
            if cv2.waitKey(1) & 0xFF == 27:
                break

        cap.release()

if __name__ == '__main__':
    extract_skeleton_from_webcam()
