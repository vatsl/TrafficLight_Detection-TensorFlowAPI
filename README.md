[//]: # (Image References)
[left0000]: ./examples/left0000.jpg
[left0003]: ./examples/left0003.jpg
[left0011]: ./examples/left0011.jpg
[left0027]: ./examples/left0027.jpg
[left0140]: ./examples/left0140.jpg
[left0701]: ./examples/left0701.jpg

[real0000]: ./examples/real0000.png
[real0140]: ./examples/real0140.png
[real0701]: ./examples/real0701.png
[sim0003]: ./examples/sim0003.png
[sim0011]: ./examples/sim0011.png
[sim0027]: ./examples/sim0027.png

# Traffic Light Detection and Classification with TensorFlow Object Detection API
---

#### A brief introduction to the project is available [here](https://medium.com/@Vatsal410/traffic-light-detection-tensorflow-api-c75fdbadac62)

---

AWS AMI with all the software dependencies like TensorFlow and Anaconda (in the community AMIs) - `udacity-carnd-advanced-deep-learning`

### Get the dataset

[Drive location](https://drive.google.com/file/d/0B-Eiyn-CUQtxdUZWMkFfQzdObUE/view?usp=sharing)

### Get the models

Do `git clone https://github.com/tensorflow/models.git` inside the tensorflow directory

Follow the instructions at [this page](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md) for installing some simple dependencies.

**All the files have to be kept inside the `tensorflow/models/research/` directory - data/, config/, data_conversion python files, .record files and utilitites/ ,etc.**


### Location of pre-trained models:
[pre-trained models zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)

Download the required model tar.gz files and untar them into `/tensorflow/models/research/` directory with `tar -xvzf name_of_tar_file`.

### Creating TFRecord files:

`python data_conversion_udacity_sim.py --output_path sim_data.record`

`python data_conversion_udacity_real.py --output_path real_data.record`

---

## Commands for training the models and saving the weights for inference.

## Using Faster-RCNN model

### For Simulator Data

#### Training

`python object_detection/train.py --pipeline_config_path=config/faster_rcnn-traffic-udacity_sim.config --train_dir=data/sim_training_data/sim_data_capture`

#### Saving for Inference

`python object_detection/export_inference_graph.py --pipeline_config_path=config/faster_rcnn-traffic-udacity_sim.config --trained_checkpoint_prefix=data/sim_training_data/sim_data_capture/model.ckpt-5000 --output_directory=frozen_sim/`


### For Real Data

#### Training

`python object_detection/train.py --pipeline_config_path=config/faster_rcnn-traffic_udacity_real.config --train_dir=data/real_training_data`

#### Saving for Inference

`python object_detection/export_inference_graph.py --pipeline_config_path=config/faster_rcnn-traffic_udacity_real.config --trained_checkpoint_prefix=data/real_training_data/model.ckpt-10000 --output_directory=frozen_real/`

---

## Using Inception SSD v2

### For Simulator Data

#### Training

`python object_detection/train.py --pipeline_config_path=config/ssd_inception-traffic-udacity_sim.config --train_dir=data/sim_training_data/sim_data_capture`

#### Saving for Inference

`python object_detection/export_inference_graph.py --pipeline_config_path=config/ssd_inception-traffic-udacity_sim.config --trained_checkpoint_prefix=data/sim_training_data/sim_data_capture/model.ckpt-5000 --output_directory=frozen_models/frozen_sim_inception/`


### For Real Data

#### Training

`python object_detection/train.py --pipeline_config_path=config/ssd_inception-traffic_udacity_real.config --train_dir=data/real_training_data`

#### Saving for Inference

`python object_detection/export_inference_graph.py --pipeline_config_path=config/ssd_inception-traffic_udacity_real.config --trained_checkpoint_prefix=data/real_training_data/model.ckpt-10000 --output_directory=frozen_models/frozen_real_inception/`

---

## Using MobileNet SSD v1
(Due to some unknown reasons the model gets trained but does not save for inference. Ignoring this for now.)

### For Simulator Data

#### Training

`python object_detection/train.py --pipeline_config_path=config/ssd_mobilenet-traffic-udacity_sim.config --train_dir=data/sim_training_data/sim_data_capture`

#### Saving for Inference

`python object_detection/export_inference_graph.py --pipeline_config_path=config/ssd_mobilenet-traffic-udacity_sim.config --trained_checkpoint_prefix=data/sim_training_data/sim_data_capture/model.ckpt-5000 --output_directory=frozen_models/frozen_sim_mobile/`


### For Real Data

#### Training

`python object_detection/train.py --pipeline_config_path=config/ssd_mobilenet-traffic_udacity_real.config --train_dir=data/real_training_data`

#### Saving for Inference

`python object_detection/export_inference_graph.py --pipeline_config_path=config/ssd_mobilenet-traffic_udacity_real.config --trained_checkpoint_prefix=data/real_training_data/model.ckpt-10000 --output_directory=frozen_models/frozen_real_mobile/`

---

**Inference results can be viewed using the TrafficLightDetection-Inference.ipynb or .html files.**

### Camera Image and Model's Detections      
![alt-text][left0000]
![alt-text][real0000]

![alt-text][left0140]
![alt-text][real0140]

![alt-text][left0701]
![alt-text][real0701]

![alt-text][left0003]
![alt-text][sim0003]

![alt-text][left0011]
![alt-text][sim0011]

![alt-text][left0027]
![alt-text][sim0027]

---

#### Some useful links

- [Uploading/Downloading files between AWS and GoogleDrive](http://olivermarshall.net/how-to-upload-a-file-to-google-drive-from-the-command-line/)

- [Using Jupyter notebooks with AWS](https://medium.com/towards-data-science/setting-up-and-using-jupyter-notebooks-on-aws-61a9648db6c5)
