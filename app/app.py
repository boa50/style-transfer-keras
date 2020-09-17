# from PIL import Image

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import vgg19

################## Para rodar na GPU ####################
from tensorflow.compat.v1 import ConfigProto, Session

config = ConfigProto()
config.gpu_options.allow_growth = True
session = Session(config=config)
#########################################################

content_image_path = keras.utils.get_file('paris.jpg', 'https://i.imgur.com/F28w3Ac.jpg')
style_image_path = keras.utils.get_file('starry_night.jpg', 'https://i.imgur.com/9ooB60I.jpg')
result_prefix = 'paris_generated'

total_variation_weight = 1e-6

# style_weight = 1e-6
# content_weight = 2.5e-8
style_weight = 1e-2
content_weight = 1e3
# style_weight = 1e6
# content_weight = 1

width, height = keras.preprocessing.image.load_img(content_image_path).size
img_nrows = 128
img_ncols = int(width * img_nrows / height)

# img_content = Image.open(content_image_path)
# img_style = Image.open(style_image_path)

# plt.imshow(img_content)
# plt.show()
# plt.imshow(img_style)
# plt.show()

def preprocess_image(image_path):
    img = keras.preprocessing.image.load_img(
        image_path, target_size=(img_nrows, img_ncols)
    )
    img = keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return tf.convert_to_tensor(img)

def deprocess_image(x):
    x = x.reshape((img_nrows, img_ncols, 3))
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype("uint8")
    return x

def gram_matrix(x):
    x = tf.transpose(x, (2, 0, 1))
    features = tf.reshape(x, (tf.shape(x)[0], -1))
    gram = tf.matmul(features, tf.transpose(features))
    return gram

def style_loss(style, combination):
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = img_nrows * img_ncols
    return tf.reduce_sum(tf.square(S - C)) / (4.0 * (channels ** 2) * (size ** 2))

def content_loss(base, combination):
    return tf.reduce_sum(tf.square(combination - base))

def total_variation_loss(x):
    a = tf.square(
        x[:, : img_nrows - 1, : img_ncols - 1, :] - x[:, 1:, : img_ncols - 1, :]
    )
    b = tf.square(
        x[:, : img_nrows - 1, : img_ncols - 1, :] - x[:, : img_nrows - 1, 1:, :]
    )
    return tf.reduce_sum(tf.pow(a + b, 1.25))

model = vgg19.VGG19(weights='imagenet', include_top=False)
model.trainable = False

outputs_dict = dict([layer.name, layer.output] for layer in model.layers)

feature_extractor = keras.Model(inputs=model.inputs, outputs=outputs_dict)

style_layer_names = [
    "block1_conv1",
    "block2_conv1",
    "block3_conv1",
    "block4_conv1",
    "block5_conv1",
]

content_layer_name = "block5_conv2"

def compute_loss(combination_image, content_image, style_image):
    input_tensor = tf.concat(
        [content_image, style_image, combination_image], axis=0
    )
    features = feature_extractor(input_tensor)
    
    loss = tf.zeros(shape=())
    
    layer_features = features[content_layer_name]
    content_image_features = layer_features[0, :, :, :]
    combination_features = layer_features[2, :, :, :]
    loss = loss + content_weight * content_loss(
        content_image_features, combination_features
    )
    
    for layer_name in style_layer_names:
        layer_features = features[layer_name]
        style_reference_features = layer_features[1, :, :, :]
        combination_features = layer_features[2, :, :, :]
        sl = style_loss(style_reference_features, combination_features)
        loss += (style_weight / len(style_layer_names)) * sl
        
    loss += total_variation_weight * total_variation_loss(combination_image)
    return loss

@tf.function
def compute_loss_and_grads(combination_image, content_image, style_image):
    with tf.GradientTape() as tape:
        loss = compute_loss(combination_image, content_image, style_image)
    grads = tape.gradient(loss, combination_image)
    return loss, grads

def generate_image(content_path, style_path, optimizer, num_iterations=4000, show_image_iterations=500, debug=False):
    content_image = preprocess_image(content_path)
    style_image = preprocess_image(style_path)
    combination_image = tf.Variable(preprocess_image(content_path))

    for i in range(1, num_iterations + 1):
        loss, grads = compute_loss_and_grads(
            combination_image, content_image, style_image
        )
        
        optimizer.apply_gradients([(grads, combination_image)])
        
        if i % show_image_iterations == 0:
            print("Iteration %d: loss=%.2f" % (i, loss))
            img = deprocess_image(combination_image.numpy())
            if debug:
                plt.imshow(img)
                plt.show()
            else:
                fname = "app/images/generated/" + result_prefix + "_at_iteration_%d.png" % i
                keras.preprocessing.image.save_img(fname, img)
            

# optimizer = keras.optimizers.SGD(
    # keras.optimizers.schedules.ExponentialDecay(
    #     initial_learning_rate=100.0, decay_steps=100, decay_rate=0.96
    # )
# )
optimizer = keras.optimizers.Adam(
    keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=10.0, decay_steps=100, decay_rate=0.96
    )
)
# optimizer = keras.optimizers.Adam(learning_rate=10.0)
# optimizer = keras.optimizers.Adam(learning_rate=0.003)

generate_image(content_image_path, style_image_path, optimizer, show_image_iterations=1000, debug=True)