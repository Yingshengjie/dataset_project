import rclpy
from rclpy.node import Node
from ultralytics import YOLO
import cv2
import time
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose


class YoloDetectNode(Node):
    def __init__(self):
        super().__init__("yolo_detect_node")
        self.pub = self.create_publisher(Detection2DArray, "/yolo/detections", 10)

        # 加载训练好的模型，best.pt放在同目录yolo_exp下
        self.model = YOLO("best.pt")
        self.conf_thresh = 0.5
        self.iou_thresh = 0.45

        # USB摄像头
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)  # 关闭自动曝光
        self.cap.set(cv2.CAP_PROP_EXPOSURE, 80)        # 设置手动曝光

        self.get_logger().info("YOLO?ROS2检测节点启动")
        self.run_loop()

    def run_loop(self):
        while rclpy.ok():
            t0 = time.time()
            ret, frame = self.cap.read()
            if not ret:
                self.get_logger().warn("摄像头读取失败")
                time.sleep(0.05)
                continue


            results = self.model(frame,
                     conf=self.conf_thresh,
                     iou=self.iou_thresh,
                     verbose=False,
                     stream=True)
            det_array = Detection2DArray()

            for res in results:
                for box in res.boxes:
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    cls_name = self.model.names[cls_id]

                    # 在图像上绘制检测框、类别、置信度
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{cls_name} {conf:.2f}",
                                (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 255, 0), 2)

                    # 封装ROS2 Detection2D消息
                    det = Detection2D()
                    hyp = ObjectHypothesisWithPose()
                    hyp.hypothesis.class_id = str(cls_id)
                    hyp.hypothesis.score = conf
                    det.results.append(hyp)

                    det.bbox.center.position.x = (x1 + x2) / 2.0
                    det.bbox.center.position.y = (y1 + y2) / 2.0
                    det.bbox.size_x = float(x2 - x1)
                    det.bbox.size_y = float(y2 - y1)
                    det_array.detections.append(det)

            self.pub.publish(det_array)

            fps = 1.0 / (time.time() - t0)
            fps = 1.0 / cost if cost > 1e-6 else 0
            cv2.putText(frame, f"FPS:{fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            cv2.imshow("Jetson Detect", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            rclpy.spin_once(self, timeout_sec=0)
        self.cap.release()
        cv2.destroyAllWindows()


def main():
    rclpy.init()
    node = YoloDetectNode()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()