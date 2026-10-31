# Models

This folder is where `iqkit` looks for the YuNet face-detection model. The file is not stored in the
repository. Download it once:

```
curl -L -o models/face_detection_yunet_2023mar.onnx \
  https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
```

YuNet comes from the [OpenCV Zoo](https://github.com/opencv/opencv_zoo) and is MIT licensed.
You can keep the file elsewhere and pass `--face-model PATH` or set `IQKIT_YUNET_MODEL`.
