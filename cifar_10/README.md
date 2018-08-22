Download the CIFAR-10 dataset and generate TFRecord files using the provided script. The script and associated command below will download the CIFAR-10 dataset and then generate a TFRecord for the training, validation, and evaluation datasets.

git$ python generate_cifar10_tfrecords.py --data-dir=${PWD}/cifar-10-data

After running the command above, you should see the following files in the --data-dir (ls -R cifar-10-data):
    train.tfrecords
    validation.tfrecords
    eval.tfrecords
    
Model Structure:

CNN -> several residual blocks -> full connection 

accuracy: 
3 residual blocks and each block has 2 residual unit: 48%
3 residual blocks and each block has 10 residual unit