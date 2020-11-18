# ros_cvdnn
Deep learning inference nodes using OpenCV DNN module.

## Node types

* [Classification](https://github.com/neka-nat/ros_cvdnn/tree/master/cvdnn_classification)

## Testing

Publish the test image.

```
rosrun image_publisher image_publisher src/ros_cvdnn/test_images/space_shuttle.jpg
```

Launch the ros node.

```
roslaunch cvdnn_classification cvdnn_classification.launch input_topic_name:=<image topic name>
```

Show the result.

```
rosrun image_view image_view image:=/overlay
```