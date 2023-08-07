from LSTM_model import HandGesture
import cv2
import mediapipe as mp
import torch
from torch.nn.functional import softmax
import numpy as np
import pandas as pd
import os.path as osp
import imutils
import time

def UniformlySample(num_frames, clip_len):
    """Uniformly sample indices for training clips.
    Args:
        num_frames (int): The number of frames.
        clip_len (int): The length of the clip.
    """
    num_clips = 1
    p_interval = (1,1)
    allinds = []
    for clip_idx in range(num_clips):
        old_num_frames = num_frames
        pi = p_interval
        ratio = np.random.rand() * (pi[1] - pi[0]) + pi[0]
        num_frames = int(ratio * num_frames)
        off = np.random.randint(old_num_frames - num_frames + 1)

        if num_frames < clip_len:
            start = np.random.randint(0, num_frames)
            inds = np.arange(start, start + clip_len)
        elif clip_len <= num_frames < 2 * clip_len:
            basic = np.arange(clip_len)
            inds = np.random.choice(
                clip_len + 1, num_frames - clip_len, replace=False)
            offset = np.zeros(clip_len + 1, dtype=np.int64)
            offset[inds] = 1
            offset = np.cumsum(offset)
            inds = basic + offset[:-1]
        else:
            bids = np.array(
                [i * num_frames // clip_len for i in range(clip_len + 1)])
            bsize = np.diff(bids)
            bst = bids[:clip_len]
            offset = np.random.randint(bsize)
            inds = bst + offset

        inds = inds + off
        num_frames = old_num_frames

        allinds.append(inds)

    return np.concatenate(allinds)


def get_kp(image):
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands
    h, w, _ = image.shape
    with mp_hands.Hands(
            model_complexity=1,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands:
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        skeleton = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:

                hand_list = list(hand_landmarks.landmark)
                instance = []
                for keypoint in hand_list:
                    instance.append([keypoint.x * w, keypoint.y * h, keypoint.z])

                skeleton.append(np.array(instance).flatten())

                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
            num_hand = len(results.multi_hand_landmarks)
            if num_hand == 1:
                skeleton.append(np.zeros(21 * 3))
            skeleton_arr = np.array(skeleton).flatten()
        else:
            skeleton_arr = np.zeros(21 * 2 * 3)
    return skeleton_arr.flatten(), image


class_name = ['RIGHT', 'LEFT', 'OK', 'UP', 'DOWN', 'PAUSE', 'ZOOM IN', 'ZOOM OUT', 'NO', 'NO ACTION']
predict_step = 10
if __name__ == "__main__":
    Model = HandGesture(input_dim=126, hidden_dim=256, n_layers=5, num_classes=10)
    Model.eval()
    Model.to('cuda')
    Model.load_state_dict(torch.load('Model_2.pth'))
    out = [[0, 0,  0, 0, 0, 0, 0,  0, 0, 0]]
    cap = cv2.VideoCapture(0)
    stack = []
    class_index = 0
    start = 0

    while True:
        flag, image = cap.read()
        skeleton_arr, image_vis = get_kp(image)
        stack.append(skeleton_arr)
        if len(stack) == 30:
            inds = UniformlySample(30, 15)

            input = np.array(stack)
            input = input[inds]
            input = torch.Tensor(input).float()

            out = Model(input[None].to('cuda'))
            out = softmax(out).to('cpu')
            out = out.detach().numpy()
            class_index = np.argmax(out)
            print(class_index)
            print(out)
            stack = []

        image_vis = cv2.flip(image_vis, 1)
        image_flip = imutils.resize(image_vis, width=800)
        fps = int(1/(time.time() - start))
        start = time.time()

        image_flip = cv2.putText(image_flip, 'Class {} scorce {}'.format(class_name[class_index], out[0][class_index]),
                    (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        image_flip = cv2.putText(image_flip, 'fps: {}'.format(fps),
                                 (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.imshow('MediaPipe Hands', image_flip)
        if cv2.waitKey(1) & 0xFF == 27:
            break