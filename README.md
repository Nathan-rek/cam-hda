# TensorFlow - Model Conversion and Edge TPU 

## Create Your Own EdgeTPU-Compatible `.tflite` Model

**Creating a TensorFlow Lite Model**:  
   I used the tutorial [Retrain SSD MobileNet V1 Object Detector on Google Colab (TF1)](https://coral.ai/docs/edgetpu/retrain-detection/) to create my EdgeTPU-compatible model. I faced issues with two Google Colab tutorials, but this one worked well for me.
   
**Video Tutorials**:
    - [DIY Custom Object Detection Model via Transfer Learning (TensorFlow Lite Edge TPU)](https://www.youtube.com/watch?v=OJ6IXygqgME&t=217s) by [Edgecate](https://www.youtube.com/@edgecate)
    
**Command to Create a TFRecord**:

If you need to create a TFRecord, use the following command:

    python3 object_detection/dataset_tools/create_pet_tf_record.py --label_map_path=/tensorflow/models/research/learn_pet/pet/pet_label_map.pbtxt --data_dir=/tensorflow/models/research/learn_pet/pet/ --output_dir=/tensorflow/models/research/learn_pet//pet/

After model train convert it for Edge TPU

   
    ./convert_checkpoint_to_edgetpu_tflite.sh --checkpoint_num 500
	

4. **Inspecting Your Model**:
    - You can use [Netron](https://netron.app/) to check the structure of your model.

5. **Important Notes**:
    - Be cautious: for Docker, you need an AMD64 architecture. **aarch64** and **armv7i** architectures cannot build the Docker image provided by google-coral.

## YOLO Model Conversion

If you want to convert a YOLO model to an EdgeTPU-compatible `.tflite` model, you need to apply **int8 quantization**. Here's what I've learned so far:

- I tried converting a YOLOv8 model to an EdgeTPU-compatible `.tflite` model for use with the TensorFlow Lite interpreter optimized for Coral. While I encountered some issues, I believe it's possible to make it work.  
- You can check out this video on [Coral TPU YOLOv5s](https://www.youtube.com/watch?v=D9IExho8pwo) for live object detection using a YOLO model on EdgeTPU.
  
For converting a YOLOv5 model, the following GitHub repository may help:

- [YOLOv5 Conversion Repository](https://github.com/zldrobit/yolov5)

## Helpful Resources

### [DAVID NYARKO](https://github.com/DAVIDNYARKO123)
- [edge-tpu-silva GitHub Repository](https://github.com/DAVIDNYARKO123/edge-tpu-silva)

### [Ultralytics Documentation](https://docs.ultralytics.com/fr/modes/export/)

- [Coral Edge TPU on Raspberry Pi with Ultralytics YOLO11 🚀](https://docs.ultralytics.com/fr/guides/coral-edge-tpu-on-raspberry-pi/)

## Coral Resources

- [Get Started with the USB Accelerator](https://coral.ai/docs/accelerator/get-started)
  
### Additional Coral Resources:

- [Google-Coral GitHub](https://github.com/google-coral/examples-camera)
    - [Raspberry Pi Camera Examples](https://github.com/google-coral/examples-camera/tree/master/raspicam)
  
- [Edge TPU Compiler Documentation](https://coral.ai/docs/edgetpu/compiler/)

---

This README provides links to essential resources for converting TensorFlow models to be EdgeTPU-compatible, as well as setup guides for various tools. By following these steps and using the resources linked above, you can create your own object detection models and run them efficiently on Coral's EdgeTPU devices.
