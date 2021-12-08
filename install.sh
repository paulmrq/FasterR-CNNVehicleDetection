# create and active a VENV
python3 -m venv venv
source ./venv/bin/activate

# install requirements
pip install -r requirements.txt

# download pretrained VGG16 weights.
wget https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5