# yolov7-Keypoint Extraction- from pose.

### Steps to run Code
- If you are using google colab then you will first need to mount the drive with mentioned command first,
STEP 1:
```
from google.colab import drive
drive.mount("/content/drive")
```
- STEP 2: Clone the repository.
```
git clone https://github.com/abhinavchoudhary2991/yolov7pose1.git
```

- STEP 3: Change Directory: Goto the cloned folder.
```
cd <filename>
```


- STEP 4: Install requirements with mentioned command below.

```
pip install -r requirements.txt
```

- STEP 5: Download yolov7 pose estimation weights 
```
"!curl -L https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6-pose.pt -o yolov7-w6-pose.pt"
```
or

from [link](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6-pose.pt) and move them to the working directory {yolov7-pose-estimation}

- STEP 6: Create 2 new folders "input" and "output". Upload any single video inside "input" folder.

- Run the code with mentioned command below.
```
!python keypoint.py 
```


python pose-estimate.py


#For CPU
python pose-estimate.py --source "your custom video.mp4" --device cpu

#For GPU
python pose-estimate.py --source "your custom video.mp4" --device 0




