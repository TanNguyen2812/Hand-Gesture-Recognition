# Hand-Getsture-Recognition
## Demo:  
[![Demo Video]
(https://img.youtube.com/vi/JLMbpiywVxQ/maxresdefault.jpg)]
(https://www.youtube.com/watch?v=JLMbpiywVxQ)



## Pipeline:  
* **Extract Keypoints:** [Mediapipe](https://developers.google.com/mediapipe)  
* **Recognition:** LSTM

![Picture1](https://github.com/TanNguyen2812/Hand-Getsture-Recognition/assets/141646071/f1026456-74c7-49d5-a4d6-181359f24e38)

## Dataset:  

![Screenshot 2023-08-07 233539](https://github.com/TanNguyen2812/Hand-Getsture-Recognition/assets/141646071/c6a2e9c9-08a3-4993-8faa-24e9c4f1e6f0)

* Collect video by webcam of laptop
* Use Mediapipe model to detect Hand Keypoints, then save to a pickle file (file.pkl)
* **10 classes**: UP, DOWN , LEFT, RIGTH, OK,NO, PAUSE,ZOOM IN, ZOOM OUT AND NO ACTION
* **684 video**, Train(70%), Validation(15%), Test(15%)
## Data pre-processing: 
Uniform Sampling: 
* Spliting the video into T segments equal length.  
* Choosing one frame from each segment randomly.
![image](https://github.com/TanNguyen2812/Hand-Getsture-Recognition/assets/141646071/be9a19c7-7295-4287-8d6e-c7a9334576bf)

## Training Config: 
* Num epochs = 200
* Learning rate init = 0.01, Learning rate scheduler: Step at 100 epoch, decay=0.5
* Batch size = 15
* Optimizer: SGD with momentum=0.9 and Nesterov=True
* Loss: CrossEntropy
 ![image](https://github.com/TanNguyen2812/Hand-Getsture-Recognition/assets/141646071/ab070d85-b2b2-4d04-b398-ee815fd05770)
## Evaluation: 
* Accuracy (Top1Acc) = 97%
![image](https://github.com/TanNguyen2812/Hand-Getsture-Recognition/assets/141646071/bde733c2-ced2-4dc9-af3d-ec4b39b69b45)
* Confusion Matrix:  
![image](https://github.com/TanNguyen2812/Hand-Getsture-Recognition/assets/141646071/187b0a25-d7aa-44ea-9b4f-3862a0aad50b)


