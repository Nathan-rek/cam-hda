# FR

Le projet *"Searching"* est un projet qui nait dans le cadre d'une pratique artistique. Dans le cadre de ce projet j'utilise un rasberry pi un pi camera et des modèles tensorflow, leur utilisation est documenté ici. De par son cadre artistique la documentation est orienté vers une utilisation spécifique de ces outils.

Searching consiste en la production d'une série d'épisodes de court film. Ces films sont réalisées par des personnes différentes mais ont comme point commun l'utilisation d'une caméra doté de reconnaissance de patterns.


---


## Matériel Utilisé

- **Raspberry Pi 4** : Flasché en 32 bits. Le choix du 32 bits est lié à des contraintes d’utilisation de la librairie PiCamera et OpenCV. En effet, MMAL (Multi-Media Abstraction Layer) n'est pas pris en charge en 64 bits. Pour plus d'informations, vous pouvez consulter ce post du forum Raspberry Pi concernant l'erreur [MMAL 64-bit support](https://github.com/raspberrypi/userland/issues/688).
  
- **PiCamera V2** : Résolution vidéo de 1080p à 47 fps, 1640 x 1232 à 41 fps et 640 x 480 à 206 fps, ce qui est important pour le workflow de capture vidéo. [Spécifications matérielles](https://www.raspberrypi.com/documentation/accessories/camera.html).

- **Google Coral USB Accelerator** : Utilisé pour accélérer les modèles TensorFlow Lite via le Edge TPU (Tensor Processing Unit).

### Installation du Raspberry Pi OS (32-bit)

1. Téléchargez l’image de Raspberry Pi OS (version 32-bit) et flashez l’image sur une carte SD en utilisant un outil comme [Raspberry Pi Imager](https://www.raspberrypi.org/software/).
2. Insérez la carte SD dans le Raspberry Pi et démarrez le système.

Installation de Pycoral avec USB Accelerator

- [Get started with the USB Accelerator](https://coral.ai/docs/accelerator/get-started). Possibiliter d'instaler aussi via la commande:

```
pip install --extra-index-url https://google-coral.github.io/py-repo/ pycoral~=2.0
```

Si sudo apt-get install python3-pycoral ne fonction pas 

### Creation du modèle:

## Entraînement du Modèle

J'ai utilisé la documentation officielle de Coral pour créer un modèle de détection d'objets optimisé pour le Edge TPU. Voici les étapes :

1. Suivez le tutoriel [Retrain SSD MobileNet V1 Object Detector on Google Colab (TF1)](https://coral.ai/docs/edgetpu/retrain-detection/) pour créer votre modèle.


**Command to Create a TFRecord**:

If you need to create a TFRecord, use the following command:

    python3 object_detection/dataset_tools/create_pet_tf_record.py --label_map_path=/tensorflow/models/research/learn_pet/pet/pet_label_map.pbtxt --data_dir=/tensorflow/models/research/learn_pet/pet/ --output_dir=/tensorflow/models/research/learn_pet//pet/

  
2. **Remarque importante** : Le processus d'entraînement nécessite Docker configuré pour l'architecture **AMD64**, ce qui n'est pas compatible directement avec un Raspberry Pi (qui utilise l'architecture **armv7i**). J'ai donc installé Docker sur Windows et utilisé un émulateur Linux (comme **WSL2**) pour réaliser l'entraînement.

3. Pour plus d'informations sur l'utilisation de Docker et la création de modèles, je recommande cette vidéo de [Edgecate:**DIY Custom Object Detection Model via Transfer Learning (TensorFlow Lite Edge TPU)](https://www.youtube.com/watch?v=OJ6IXygqgME&t=850s)**.

4. **Alternative avec Google Colab** : Vous pouvez également utiliser Google Colab pour entraîner votre modèle en utilisant ce [notebook Google Colab](https://colab.research.google.com/github/google-coral/tutorials/blob/master/retrain_ssdlite_mobiledet_qat_tf1.ipynb#scrollTo=jcApdURAK28f).

### Utilisation du modèle

1. Afin de réaliser mes vidéos en meilleure résolution et avec un bon taux de FPS, j'utilise deux scripts.
- Le premier fait un enregistrement en direct de la PiCamera avec OpenCV en 1280x720 à 16 FPS, tout en effectuant simultanément une visualisation de la détection en 640x420. Toutefois, cette dernière ne contient pas les boîtes de détection sur la vidéo enregistrée en 1280x720, ce qui permet de gagner en FPS. Ce script génère également un tableau CSV qui sera ensuite utilisé pour la génération du son de la vidéo. Il sert aussi à nommer les fichiers en fonction de la dernière frame enregistrée avant. Cela 
permet de créer une séquence de capture vidéo.
  - Ce premier script est inspiré d'une version modifiée du script [TFLite_detection_webcam.py](https://github.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi) du dépôt **TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi** d'EdjeElectronics.
tps://github.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi) de EdjeElectronics
- Le deuxième script compiles les vidéos à l'aide de ffmpeg et les analysent en dessinant les boites englobantes.


## Ressources Utiles

Voici une liste de ressources qui peuvent vous aider tout au long du projet.

### Tutoriels pour la création de modèles TFLite

- [Retrain SSD MobileNet V1 Object Detector on Google Colab (TF1)](https://coral.ai/docs/edgetpu/retrain-detection/) – Tutoriel officiel de Coral pour entraîner un modèle de détection d'objets sur Google Colab.
- [DIY Custom Object Detection Model via Transfer Learning (TensorFlow Lite Edge TPU)](https://www.youtube.com/watch?v=OJ6IXygqgME&t=850s) – Tutoriel vidéo de [Edgecate](https://www.youtube.com/@edgecate) expliquant comment créer un modèle de détection d'objets personnalisé à l'aide de TensorFlow Lite.
- [Notebook Google Colab pour l'entraînement de modèles SSD MobileNet](https://colab.research.google.com/github/google-coral/tutorials/blob/master/retrain_ssdlite_mobiledet_qat_tf1.ipynb#scrollTo=jcApdURAK28f) – Utilisez ce notebook pour entraîner des modèles directement sur Google Colab.

### Ressources supplémentaires pour la conversion des modèles pour Edge TPU

- [Google Coral GitHub](https://github.com/google-coral/examples-camera) – Dépôt GitHub avec des exemples d'utilisation de Coral et des Raspberry Pi Camera.
- [Edge TPU Compiler Documentation](https://coral.ai/docs/edgetpu/compiler/) – Documentation officielle pour compiler un modèle TensorFlow Lite pour Edge TPU.

### Dépôts GitHub utiles

- [Google Coral Edge TPU Examples](https://github.com/google-coral/examples-camera) – Exemples d'utilisation du Google Coral USB Accelerator avec des Raspberry Pi et des caméras.

### Documentation TensorFlow

- [Documentation officielle de TensorFlow Lite](https://www.tensorflow.org/lite) – Guide complet sur TensorFlow Lite, y compris la conversion et l'optimisation des modèles.

### Utilisation de Yolo

L'utilisation d'un modèle Yolo est aussi une solution, l'ayant envisagé un moment voici les dépôts et docs que j'ai trouvé pour utiliser ces modèles.
- [YOLOv5 Conversion Guide](https://docs.ultralytics.com/fr/modes/export/) – Tutoriel pour la conversion des modèles YOLOv5 en modèles TensorFlow Lite optimisés pour le Edge TPU.
- [edge-tpu-silva GitHub Repository](https://github.com/DAVIDNYARKO123/edge-tpu-silva) – Un autre dépôt pour des exemples de projets utilisant le Coral USB Accelerator.

- [Ultralytics Documentation](https://docs.ultralytics.com/fr/modes/export/)

- [Coral Edge TPU on Raspberry Pi with Ultralytics YOLO11 🚀](https://docs.ultralytics.com/fr/guides/coral-edge-tpu-on-raspberry-pi/)

## Conversion du modèle YOLO

Si vous souhaitez convertir un modèle YOLO en un modèle `.tflite` compatible avec EdgeTPU, vous devez appliquer la **quantification int8**. Voici ce que j'ai appris jusqu'à présent :

- J'ai essayé de convertir un modèle YOLOv8 en un modèle `.tflite` compatible avec EdgeTPU pour l'utiliser avec l'interpréteur TensorFlow Lite optimisé pour Coral. Bien que j'aie rencontré quelques problèmes, je pense qu'il est possible de le faire fonctionner.
- Vous pouvez consulter cette vidéo sur [Coral TPU YOLOv5s](https://www.youtube.com/watch?v=D9IExho8pwo) pour la détection d'objets en direct utilisant un modèle YOLO sur EdgeTPU.

Pour convertir un modèle YOLOv5, le dépôt GitHub suivant peut être utile :

- [Dépôt de conversion YOLOv5](https://github.com/zldrobit/yolov5)


## Problèmes et Dépannage

### Problème de compatibilité avec Docker et Raspberry Pi

Lors de la création du modèle avec Docker, j'ai rencontré des problèmes de compatibilité liés à l'architecture. Docker sur Raspberry Pi utilise **armv7i**, tandis que certains outils de création de modèles (comme ceux utilisés pour l'Edge TPU) nécessitent une architecture **AMD64**. Pour contourner cette limitation, j'ai utilisé Docker sur un système Windows via WSL2 (Windows Subsystem for Linux).

### Problème de compatibilité MMAL en 64-bit

Le problème d'incompatibilité avec MMAL (Multi-Media Abstraction Layer) en 64 bits empêche l'utilisation de certaines fonctionnalités de la caméra Pi sur des systèmes 64-bit. Le Raspberry Pi OS 32-bit est nécessaire pour garantir la compatibilité avec la caméra Pi et les bibliothèques comme OpenCV et PiCamera. Vous pouvez consulter plus de détails dans le post du forum Raspberry Pi [MMAL 64-bit support](https://github.com/raspberrypi/userland/issues/688).

### Problème de compatibilité avec picamera module 3 et Raspberry Pi 5 et python 3.9

la lib picamera n'est pas possible à installer dans un environement virtuel python 3.9 

```
https://github.com/raspberrypi/picamera2/issues/446
https://github.com/raspberrypi/picamera2/issues/503
```

Or, il faut un environnement virtuel pour installer **PyCoral** qui fonctionne sur Python 3.9.  
**Picamera2** nécessite **libcamera**, qui ne peut être installé que via **apt** (`sudo apt install`). Cela empêche son installation dans un environnement virtuel Conda.  
**JungLearnBot** propose un détournement possible sur Python 3.11.  
- [Readme.RPi5.cpu.picam.qt.md](https://github.com/JungLearnBot/RPi5_yolov8/blob/main/Readme.RPi5.cpu.picam.qt.md)

### Problème de compatibilité entre OpenCV et libcamera (Picamera2)

**OpenCV VideoCapture** n'est pas compatible avec **Pi Camera Module 3**. Cela est dû à la façon dont **libcamera** gère les flux vidéo, ce qui pose des problèmes d'accès direct à la caméra avec OpenCV.  
Même après l'installation de **Picamera2**, la fonction **cv2.imshow()** de **OpenCV** ne fonctionne pas correctement. L'application se fige lorsque tu essaies d'afficher le flux de la caméra.  
En réponse, **JungLearnBot** propose d'opter pour **Qt** pour la visualisation des flux vidéo, mais il y a aussi des problèmes de compatibilité avec OpenCV sous **Pi OS** (ce qui nécessite l'installation de **opencv-python-headless** pour éviter les conflits).

Utiliser **Picamera2** à la place de **OpenCV VideoCapture**, mais cela nécessite de gérer les installations de **libcamera** et de manipuler les bibliothèques globales via des détournements possibles sur Python 3.11.

Sur **Pi Camera Module 3**, utiliser **Qt** pour la visualisation vidéo, car **cv2.imshow** pose des problèmes avec **Picamera2** et installer **opencv-python-headless** pour éviter les conflits avec **Qt** sur **Pi OS**.


---


## Ressources utiles

### [DAVID NYARKO](https://github.com/DAVIDNYARKO123)
- [Dépôt GitHub edge-tpu-silva](https://github.com/DAVIDNYARKO123/edge-tpu-silva)

- [Edje Electronics / TensorFlow Lite pour la détection d'objets sur Android et Raspberry Pi](https://github.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi/tree/master)

### [Documentation Ultralytics](https://docs.ultralytics.com/fr/modes/export/)

- [Coral Edge TPU sur Raspberry Pi avec Ultralytics YOLO11 🚀](https://docs.ultralytics.com/fr/guides/coral-edge-tpu-on-raspberry-pi/)

### Ressources Coral

- [Commencer avec l'accélérateur USB](https://coral.ai/docs/accelerator/get-started)
  
#### Ressources Coral supplémentaires :

- [Google-Coral GitHub](https://github.com/google-coral/examples-camera)
    - [Exemples de caméra Raspberry Pi](https://github.com/google-coral/examples-camera/tree/master/raspicam)
  
- [Documentation du compilateur Edge TPU](https://coral.ai/docs/edgetpu/compiler/)

### Pour PI5 and picamera3: [JungLearnBot](https://github.com/JungLearnBot)
- [RPi5_yolov8](https://github.com/JungLearnBot/RPi5_yolov8)


### Création de la coque: [Kevin McAleer](https://www.youtube.com/@kevinmcaleer28)
- [Is the Raspberry Pi High-Quality Camera worth it? & Build your own Camera
](https://www.youtube.com/watch?v=4BEjKUK8DSQ)

---


ENG


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
    - If you use a Google Coral, plug the coral on the blue usb port.


## YOLO Model Conversion

If you want to convert a YOLO model to an EdgeTPU-compatible `.tflite` model, you need to apply **int8 quantization**. Here's what I've learned so far:

- I tried converting a YOLOv8 model to an EdgeTPU-compatible `.tflite` model for use with the TensorFlow Lite interpreter optimized for Coral. While I encountered some issues, I believe it's possible to make it work.  
- You can check out this video on [Coral TPU YOLOv5s](https://www.youtube.com/watch?v=D9IExho8pwo) for live object detection using a YOLO model on EdgeTPU.
  
For converting a YOLOv5 model, the following GitHub repository may help:

- [YOLOv5 Conversion Repository](https://github.com/zldrobit/yolov5)
