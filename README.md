This repo contrains files needed to get YOLO running on a dataset quickly and easily.

To use, you need a dataset of images, normally ~100 is a good start but this depends on the specific use case and precision needed.

# Dependancies
All files were made with google colab in mind but it should be fine to run them on any jupyter environemnt. The annotation widget requires a jupyter notebook environment to run but the other scripts could easily be adapted for any python environment

# Files descriptions
* **annotate.ipynb**: This file runs the annotation widget with which you can draw your training bounding boxes, and converts them into a format that YOLO can use for training
* **train.ipynb**: This trains the YOLO model using the output from the annotate file
* **run.ipynb**: This gives a basic example of how to use a trained model and interact with the results in python. You will likely want to adapt this for your use case.

Additional running instructions are included in the files themselves
