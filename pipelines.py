from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
import numpy as np
import tensorflow as tf
from hexgame import HexGameState


def build_conv_model(board_size):
    input_shape = (None, board_size, board_size, 2) #first dim is for batch
    # define input layers
    board_input = tf.keras.layers.Input(shape=input_shape[1:])
    player_input = tf.keras.layers.Input(shape=(1,))
    # separate channels for black and red players
    black_board = tf.keras.layers.Lambda(lambda x: x[:, :, :, 0:1])(board_input)
    red_board = tf.keras.layers.Lambda(lambda x: x[:, :, :, 1:2])(board_input)

    # convolutional layers
    conv1_black = tf.keras.layers.Conv2D(16, kernel_size=(2, 2), activation='relu',  kernel_initializer='random_uniform', padding = "same", bias_initializer='zeros')(black_board)
    conv2_black = tf.keras.layers.Conv2D(32, kernel_size=(2, 2), activation='relu', kernel_initializer='random_uniform', padding = "same", bias_initializer='zeros')(conv1_black)
    flatten_black = tf.keras.layers.Flatten()(conv1_black)#(conv2_black)

    conv1_red = tf.keras.layers.Conv2D(16, kernel_size=(2, 2), activation='relu',kernel_initializer='random_uniform', padding = "same", bias_initializer='zeros')(red_board)
    conv2_red = tf.keras.layers.Conv2D(32, kernel_size=(2, 2), activation='relu',kernel_initializer='random_uniform', padding = "same",bias_initializer='zeros')(conv1_red)
    flatten_red = tf.keras.layers.Flatten()(conv1_red)#(conv2_red)

    # concatenate the flattened convolution outputs
    concat = tf.keras.layers.Concatenate()([flatten_black, flatten_red, player_input])

    # dense layers
    dense1 = tf.keras.layers.Dense(32, activation='relu',kernel_initializer='random_uniform', bias_initializer='zeros')(concat)
    dense2 = tf.keras.layers.Dense(16, activation='relu',kernel_initializer='random_uniform', bias_initializer='zeros')(dense1)

    # output layer with softmax activation
    output = tf.keras.layers.Dense(16, activation='softmax',kernel_initializer='random_uniform', bias_initializer='zeros')(dense2)

    # define the model with board and player inputs and output layer
    model = tf.keras.models.Model(inputs=[board_input, player_input], outputs=output)

    # compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model



def hexgamestate_to_input(state_1D_array, board_size):
    """
    # Function to convert a batch of HexGameState represented as 1D array object to input tensor
    # for conv model with 2 conv channels and channel for player
    """
    input_players = state_1D_array[:,0]
    input_boards = state_1D_array[:, 1:].reshape(-1, board_size, board_size) #todo: generalize

    boards = np.stack([np.dstack((board == 1, board == 2)) for board in input_boards], axis=0)
    players = np.expand_dims(input_players, axis=1)

    return [boards, players]



def build_conv_pipeline(board_size):
    """
    Builds a pipeline for the model that uses conv2D layers with the specified bord size
    """
    model = build_conv_model(board_size)
    pipeline = Pipeline([
        ('preprocess', FunctionTransformer(hexgamestate_to_input, kw_args={'board_size': board_size})),
        ('model', model)
    ])
    return pipeline


