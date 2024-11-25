Tuto and help 

## [DAVID NYARKO](https://github.com/DAVIDNYARKO123)
- [edge-tpu-silva](https://github.com/DAVIDNYARKO123/edge-tpu-silva)

## [Ultralytics doc](https://docs.ultralytics.com/fr/modes/export/)

- [Coral Edge TPU sur un Raspberry Pi avec Ultralytics YOLO11 🚀](https://docs.ultralytics.com/fr/guides/coral-edge-tpu-on-raspberry-pi/)


## Coral

- [Get started with the USB Accelerator](https://coral.ai/docs/accelerator/get-started)
  - ### [Google-coral](https://github.com/google-coral/examples-camera)
    - [raspicam](https://github.com/google-coral/examples-camera/tree/master/raspicam)
 - ### [Edge TPU Compiler](https://coral.ai/docs/edgetpu/compiler/)

## TensorFlow

Created your own edgetpu.tflite model

-  I use this one [Retrain SSD MobileNet V1 object detector on Google Colab (TF1)](https://coral.ai/docs/edgetpu/retrain-detection/) because i meet issue with the two google colab.
  - Assiste with tutorial[DIY Custom Object Detection Model via Transfer Learning (Tensorflow Lite Edge TPU)](https://www.youtube.com/watch?v=OJ6IXygqgME&t=217s) by [Edgecate](https://www.youtube.com/@edgecate)

You can use [netron.app](https://netron.app/) for checking your model structure.

**Be careful For this Docker you need a AMD64 Architecture so aarch64 and armv7i can't build docker image**

- I try to convert yolov8 model to edgetpu.tflite model for using tfliter interpreter optimise for coral. But when i use converted model i foudn some issue but i think it's possible because this video [coral TPU yolov5s](https://www.youtube.com/watch?v=D9IExho8pwo)show live objet detection

i think this git https://github.com/zldrobit/yolov5 can help for conversion in yolo5 base model



Google colab who can create model:
- [Retrain EfficientDet-Lite object detector on Google Colab (TF2)](https://colab.research.google.com/github/google-coral/tutorials/blob/master/retrain_efficientdet_model_maker_tf2.ipynb)
- [Retrain SSDLite MobileDet object detector on Google Colab (TF1)](https://colab.research.google.com/github/google-coral/tutorials/blob/master/retrain_ssdlite_mobiledet_qat_tf1.ipynb)





 