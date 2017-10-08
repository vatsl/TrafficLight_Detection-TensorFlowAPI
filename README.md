# Traffic Light Detection and Classification with TensorFlow Object Detection API
---

AWS AMI with all the software dependencies like TensorFlow and Anaconda- udacity-carnd-advanced-deep-learning

### Get the dataset

[Drive location](https://drive.google.com/file/d/0B-Eiyn-CUQtxdUZWMkFfQzdObUE/view?usp=sharing)

### Get the models

Do `git clone https://github.com/tensorflow/models.git` inside the tensorflow directory

Follow the instructions at [this page](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md) for installing some simple dependencies.

*All the files have to be kept inside the `tensorflow/models/research/` directory - data/, config/, data_conversion python files, .record files and utilitites/ ,etc. *


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
s
`python object_detection/export_inference_graph.py --pipeline_config_path=config/faster_rcnn-traffic-udacity_sim.config --trained_checkpoint_prefix=data/sim_training_data/sim_data_capture/model.ckpt-5000 --output_directory=frozen_sim/`

Old command (ignore this): `python object_detection/train.py --pipeline_config_path=/home/ubuntu/new_data/config/faster_rcnn-traffic-udacity_sim.config --train_dir=/home/ubuntu/new_data/data/sim_training_data/sim_data_capture`


### For Real Data

#### Training

`python object_detection/train.py --pipeline_config_path=/home/ubuntu/new_data/config/faster_rcnn-traffic_udacity_real.config --train_dir=/home/ubuntu/new_data/data/real_training_data`

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

*Inference results can be viewed using the 

#### Some useful links

- [Uploading/Downloading files between AWS and GoogleDrive](http://olivermarshall.net/how-to-upload-a-file-to-google-drive-from-the-command-line/)

- [Using Jupyter notebooks with AWS](https://medium.com/towards-data-science/setting-up-and-using-jupyter-notebooks-on-aws-61a9648db6c5)
