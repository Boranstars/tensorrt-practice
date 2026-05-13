/usr/src/tensorrt/bin/trtexec \
--onnx=/home/jetson/Programs/tensorrt/tensorrt-practice/src/yolov5/models/best.onnx \
--saveEngine=/home/jetson/Programs/tensorrt/tensorrt-practice/src/yolov5/models/best.engine \
--fp16 \
--duration=30 \
--warmUp=1000 \
--profilingVerbosity=detailed