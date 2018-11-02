import torch
import torch.nn as nn
from torch.autograd import Variable
import keras.backend as K
from keras.models import *
from keras.layers import *

import torch
from torchvision.models import squeezenet1_1

from models import Net


class PytorchToKeras(object):
    def __init__(self, pModel, kModel):
        super(PytorchToKeras, self)
        self.__source_layers = []
        self.__target_layers = []
        self.pModel = pModel
        self.kModel = kModel

        K.set_learning_phase(0)

    def __retrieve_k_layers(self):

        for i, layer in enumerate(self.kModel.layers):
            if len(layer.weights) > 0:
                self.__target_layers.append(i)

    def __retrieve_p_layers(self, input_size):

        input = torch.randn(input_size)

        input = Variable(input.unsqueeze(0))

        hooks = []

        def add_hooks(module):

            def hook(module, input, output):
                if hasattr(module, "weight"):
                    self.__source_layers.append(module)

            if not isinstance(module, nn.ModuleList) and not isinstance(module,
                                                                        nn.Sequential) and module != self.pModel:
                hooks.append(module.register_forward_hook(hook))

        self.pModel.apply(add_hooks)

        self.pModel(input)
        for hook in hooks:
            hook.remove()

    def convert(self, input_size):
        self.__retrieve_k_layers()
        self.__retrieve_p_layers(input_size)

        for i, (source_layer, target_layer) in enumerate(zip(self.__source_layers, self.__target_layers)):

            weight_size = len(source_layer.weight.data.size())

            transpose_dims = []

            for i in range(weight_size):
                transpose_dims.append(weight_size - i - 1)

            self.kModel.layers[target_layer].set_weights(
                [source_layer.weight.data.numpy().transpose(transpose_dims), source_layer.bias.data.numpy()])

    def save_model(self, output_file):
        self.kModel.save(output_file)

    def save_weights(self, output_file):
        self.kModel.save_weights(output_file)


"""
We explicitly redefine the Squeezent architecture since Keras has no predefined Squeezent
"""


def squeezenet_fire_module(input, input_channel_small=16, input_channel_large=64):
    channel_axis = 3

    input = Conv2D(input_channel_small, (1, 1), padding="valid")(input)
    input = Activation("relu")(input)

    input_branch_1 = Conv2D(input_channel_large, (1, 1), padding="valid")(input)
    input_branch_1 = Activation("relu")(input_branch_1)

    input_branch_2 = Conv2D(input_channel_large, (3, 3), padding="same")(input)
    input_branch_2 = Activation("relu")(input_branch_2)

    input = concatenate([input_branch_1, input_branch_2], axis=channel_axis)

    return input


def SqueezeNet(input_shape=(224, 224, 3)):
    image_input = Input(shape=input_shape)

    network = Conv2D(64, (3, 3), strides=(2, 2), padding="valid")(image_input)
    network = Activation("relu")(network)
    network = MaxPool2D(pool_size=(3, 3), strides=(2, 2))(network)

    network = squeezenet_fire_module(input=network, input_channel_small=16, input_channel_large=64)
    network = squeezenet_fire_module(input=network, input_channel_small=16, input_channel_large=64)
    network = MaxPool2D(pool_size=(3, 3), strides=(2, 2))(network)

    network = squeezenet_fire_module(input=network, input_channel_small=32, input_channel_large=128)
    network = squeezenet_fire_module(input=network, input_channel_small=32, input_channel_large=128)
    network = MaxPool2D(pool_size=(3, 3), strides=(2, 2))(network)

    network = squeezenet_fire_module(input=network, input_channel_small=48, input_channel_large=192)
    network = squeezenet_fire_module(input=network, input_channel_small=48, input_channel_large=192)
    network = squeezenet_fire_module(input=network, input_channel_small=64, input_channel_large=256)
    network = squeezenet_fire_module(input=network, input_channel_small=64, input_channel_large=256)

    # Remove layers like Dropout and BatchNormalization, they are only needed in training
    # network = Dropout(0.5)(network)

    network = Conv2D(1000, kernel_size=(1, 1), padding="valid", name="last_conv")(network)
    network = Activation("relu")(network)

    network = GlobalAvgPool2D()(network)
    network = Activation("softmax", name="output")(network)

    input_image = image_input
    model = Model(inputs=input_image, outputs=network)

    return model

print("1")
keras_model = SqueezeNet()
print("2")
# Lucky for us, PyTorch includes a predefined Squeezenet
# pytorch_model = squeezenet1_1()
pytorch_model = Net()
print("3")

# Load the pretrained model
# pytorch_model.load_state_dict(torch.load("squeezenet.pth"))
modelCheckpoint = torch.load("m-epoch-49-21062018-033536.pth.tar", map_location='cpu')
print("4")
pytorch_model.load_state_dict(modelCheckpoint['state_dict'], strict=False)
print("5")

# Time to transfer weights
converter = PytorchToKeras(pytorch_model, keras_model)
print("6")
converter.convert((3, 224, 224))
print("7")

# Save the weights of the converted keras model for later use
converter.save_weights("squeezenet.h5")
print("8")















# import torch
# import tensorflow as tf
# import onnx
# from onnx_tf.backend import prepare
# import os
# from PIL import Image
# from torch.autograd import Variable
#
# # step 1, load pytorch model and export onnx during running.
#
#
# modelname = 'rice_modeSl'
# weightfile = 'm-epoch-49-21062018-033536.pth.tar'
# modelhandle = DIY_Model(modelname, weightfile, class_numbers)
# model = modelhandle.model
# # model.eval() # useless
# dummy_input = Variable(torch.randn(1, 3, 224, 224))  # nchw
# onnx_filename = os.path.split(weightfile)[-1] + ".onnx"
# torch.onnx.export(model, dummy_input,
#                   onnx_filename,
#                   verbose=True)
#
# # step 2, create onnx_model using tensorflow as backend. check if right and export graph.
# onnx_model = onnx.load(onnx_filename)
# tf_rep = prepare(onnx_model, strict=False)
# # install onnx-tensorflow from github，and tf_rep = prepare(onnx_model, strict=False)
# # Reference https://github.com/onnx/onnx-tensorflow/issues/167
# # tf_rep = prepare(onnx_model) # whthout strict=False leads to KeyError: 'pyfunc_0'
# image = Image.open('pants.jpg')
# # debug, here using the same input to check onnx and tf.
# output_pytorch, img_np = modelhandle.process(image)
# print('output_pytorch = {}'.format(output_pytorch))
# output_onnx_tf = tf_rep.run(img_np)
# print('output_onnx_tf = {}'.format(output_onnx_tf))
# # onnx --> tf.graph.pb
# tf_pb_path = onnx_filename + '_graph.pb'
# tf_rep.export_graph(tf_pb_path)
#
# # step 3, check if tf.pb is right.
# with tf.Graph().as_default():
#     graph_def = tf.GraphDef()
#     with open(tf_pb_path, "rb") as f:
#         graph_def.ParseFromString(f.read())
#         tf.import_graph_def(graph_def, name="")
#     with tf.Session() as sess:
#         # init = tf.initialize_all_variables()
#         init = tf.global_variables_initializer()
#         # sess.run(init)
#
#         # print all ops, check input/output tensor name.
#         # uncomment it if you donnot know io tensor names.
#         '''
#         print('-------------ops---------------------')
#         op = sess.graph.get_operations()
#         for m in op:
#             print(m.values())
#         print('-------------ops done.---------------------')
#         '''
#
#         input_x = sess.graph.get_tensor_by_name("0:0")  # input
#         outputs1 = sess.graph.get_tensor_by_name('add_1:0')  # 5
#         outputs2 = sess.graph.get_tensor_by_name('add_3:0')  # 10
#         output_tf_pb = sess.run([outputs1, outputs2], feed_dict={input_x: img_np})
#         # output_tf_pb = sess.run([outputs1, outputs2], feed_dict={input_x:np.random.randn(1, 3, 224, 224)})
#         print('output_tf_pb = {}'.format(output_tf_pb))
