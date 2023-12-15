# -*- coding: utf-8 -*-
from tensorflow.keras import backend as K
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Conv2D, Dense, Lambda
from tensorflow.keras.metrics import AUC, BinaryAccuracy
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers.schedules import ExponentialDecay

metrics = [
      BinaryAccuracy(name='accuracy'),
      AUC(name='auc')
    ]


def load_saved_model(model_path):
    """Load the saved model from the specified path."""
    return load_model(model_path)


def global_average_pooling(x):
    return K.mean(x, axis=(2, 3))


def global_average_pooling_shape(input_shape):
    return input_shape[0:2]


# best 12 layers eyepacs and 6 messidor2
def build_vgg16_GAP(n_layers=12, type_train='n', model_name='_raw', input_shape=(299, 299, 3), lr=0.01):
    """
    Atualmente o modelo mais eficiente na deteccao das retinopatias.
    Consiste de layers de uma VGG16 treinados no conjunto de
    dados imagenet

    :param n_layers: numero de layers que serao treinados
    :param type_train: tipo de treino que sera feito (normal 'n' or tranfer learning 'tl')
    :param model_name: nome do modelo base seja ele um treino normal ou uma transferencia de conhecimento
    :param input_shape: shape da imagem de entrada

    :return: modelo VGG16 com GAP
    """

    # Se o tipo de treino for normal, iremos fazer a transferencia de conhecimento apenas da ImageNet
    # Caso for FT, treinamos todas as layers do modelo com LR baixo
    # Caso for uma transferencia de conhecimento, frizar algumas layers e treinar em cima das demais com dados novos

    # ----------------------------- #
    if type_train == 'n':
        print('Generating model with ImageNet weights')
        vgg_conv = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)

        x = vgg_conv.output
        x = Lambda(global_average_pooling, output_shape=global_average_pooling_shape)(x)
        predictions = Dense(2, activation='softmax', kernel_initializer='uniform')(x)

        model = Model(inputs=vgg_conv.input, outputs=predictions)
    else:
        print('Loading the base model to '+type_train)
        model = load_model(model_name+'.hdf5')
        model_name = model_name+'_'+type_train

    # ----------------------------- #
    count = 0
    # Unlock all layers for transfer learning with specific LR
    if type_train == 'tl':
        for layer in model.layers[::-1]:
            layer.trainable = True
        lr = 0.0001
    else:
        for layer in model.layers[::-1]:

            if not isinstance(layer, Conv2D) or count > n_layers:
                layer.trainable = False
            else:
                layer.trainable = True
                count = count + 1

    lr_schedule = ExponentialDecay(
        initial_learning_rate=lr,
        decay_steps=100000,
        decay_rate=0.96,
        staircase=True)

    opt = SGD(learning_rate=lr_schedule, momentum=0.9, nesterov=True)

    model.summary()
    model.compile(loss='binary_crossentropy',
                  optimizer=opt, metrics=[metrics])

    return model, model_name
