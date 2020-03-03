# Pipeline

1. build a simple CNNs to classifiy weither a frame has a valid (frontal, large) jojo-face.
2. read from MP4 file, get all frame containing jojo-face.
3. calculate image similarity, jump over similar frames.
4. labeling face-rect with labelme.
5. train a tiny yolo-v3 to detect all other faces.