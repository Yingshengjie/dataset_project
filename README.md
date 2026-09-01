# Yolo ROS2 Detect Node
YOLO TensorRT inference node for Jetson platforms with USB camera, ROS2 detection message publishing and manual annotation snapshot tool.
## Features
- Capture frames from USB camera, run YOLO TensorRT (`best.engine`) inference
- Publish detection results as `vision_msgs/Detection2DArray` to ROS2 topic `/yolo/detections`
- Draw bounding boxes, class‑confidence, FPS and test statistics on live frame
- Manual keyboard annotation for model evaluation, save annotated snapshots locally
- Four annotation types: Correct, Missed detection, Category error, False positive
- Automatically count total tests and compute accuracy
- Preset camera exposure parameters via v4l2‑ctl

# Dependencies
## ROS2
sudo apt install ros-humble-vision-msgs
## Python packages
pip install ultralytics opencv-python

Hardware: Jetson Orin NX / Nano, USB camera at `/dev/video0`
Model: TensorRT engine file `best.engine`

# Parameters
Modify these variables inside 'main.py':
```
self.conf_thresh = 0.40          # Detection confidence threshold
self.iou_thresh = 0.55           # NMS IOU threshold
self.snapshot_dir = "/home/mash/yolo_exp/test_snapshots1"  # Snapshot output folder
self.model = YOLO("/home/mash/yolo_exp/best.engine")       # TensorRT model path
```
> Notes
> 1. Make sure `best.engine` exists and path is correct.
> 2. Adjust `/dev/video0` if your camera uses another device index.
> 3. Snapshot directory will be created automatically; change path for multiple experiment groups.

# Keyboard Shortcuts
| Key | Function |
| 1 | CORRECT, mark as correct sample, save snapshot, increment correct counter |
| 2 | WRONG: Missed detection, save snapshot |
| 3 | WRONG: Category detection error, save snapshot |
| 4 | WRONG: False positive, save snapshot |
| q | Exit program |
When pressing `1/2/3/4`:
- Snapshot named `test_001.jpg`, `test_002.jpg` … will be stored in snapshot folder
- Embedded information: test index, instant accuracy, manual annotation label
- Log will be printed in terminal

# Run
## Source ROS2 environment
source /opt/ros/humble/setup.bash
## Launch python node
python3 main.py
## Inspect ROS2 detection topic
ros2 topic echo /yolo/detections

# Snapshot Content
Each saved image contains:
- Raw frame with YOLO green bounding boxes, class name and confidence
- Real‑time FPS
- Test counter and instant accuracy
- Manual annotation label

# Log Output
- Keyboard hint printed by `print()` on startup
- ROS logger outputs annotation record:
`Test#xxx | Annotation:xxx | Current_acc:xx.x% | Saved:xxx.jpg`

# Troubleshooting
1. Camera read failed warning
   - Check device: `ls /dev/video0`
   - Permission fix: `sudo chmod 666 /dev/video0`
   - Modify camera index in `cv2.VideoCapture()` and v4l2‑ctl command.
2. Engine model loading error
   - The `.engine` file must match Jetson platform and ultralytics version, re‑export TensorRT model.
3. OpenCV window no response / key not working
   - Click the OpenCV display window to get focus before pressing keys.
4. v4l2‑ctl exposure command error
   - If your camera does not support these controls, comment out two `os.system("v4l2‑ctl ...")` lines.
5. Program crashes silently
   - The ‘try‑except’ block suppresses exceptions. Comment it temporarily to view full traceback.

# ROS2 Message
Published topic: ‘/yolo/detections’
Message type: ‘vision_msgs/msg/Detection2DArray’
Each detection includes: class id, confidence score, bounding‑box center position, box width and height. Can be subscribed by robot arm control nodes.

# Workflow
1. Start node, OpenCV camera window pops up.
2. Place target objects in camera field‑of‑view.
3. Observe YOLO detection results on screen.
4. Press `1/2/3/4` according to human judgement to save annotated snapshot. Statistics update automatically.
5. Repeat for multiple test cases; change `snapshot_dir` path for comparison between groups.
6. Press `q` to quit.

# Additional Notes
- Frame‑by‑frame debug log is commented out; uncomment related lines if you need detection box debug information.
- Accuracy formula: `accuracy = correct_count / test_count * 100%`. Only key `1` increases `correct_count`.
- Accuracy shown on snapshot is real‑time value at the moment you press annotation key.
