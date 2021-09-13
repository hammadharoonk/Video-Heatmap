# Video-Heatmap

Creating videos of aesthetic heat maps from videos of urban scenes, showing the optical flow patterns people and cars manifest, using Python and OpenCV. 

![alt text](https://github.com/hammadharoonk/Video-Heatmap/blob/main/flowpatternsvideo.gif?raw=true)

This is a study into optical flow patterns formed by the movement of large crowds of pedestrians and vehicles on a busy intersection, using simple OpenCV background subtraction. Some very interesting patterns are formed, which can show desire paths, driver habits, and pedestrian flows in the absence of adequate street infrastructure; which can help guide further street design.

Further exploration could identify different types of objects, and create separate flows for each object. Most existing image-detection models are pre-trained on frontal images so a custom model trained top-view images would be required. would appreciate any feedback or recommendations!

Video from the DUT Dataset: https://lnkd.in/d-hepWEB
Inspired by Intel Heatmap which outputs an image frame: https://github.com/intel-iot-devkit/python-cv-samples/blob/master/examples/motion-heatmap/motion-heatmap.py
My version shows the traces in real-time video, and contains adjustable trace length with a rolling First In First Out stack.
