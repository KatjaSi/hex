import tensorflow as tf
import numpy as np

from hexgame import HexGameState, HexStateManager

from pipelines import build_conv_pipeline, build_conv_model

# defining the model for anet
# define input shape
input_shape = (None, 3, 3, 2) #first dim is for batch
# define input layers
board_input = tf.keras.layers.Input(shape=input_shape[1:])
player_input = tf.keras.layers.Input(shape=(1,))
# separate channels for black and red players
black_board = tf.keras.layers.Lambda(lambda x: x[:, :, :, 0:1])(board_input)
red_board = tf.keras.layers.Lambda(lambda x: x[:, :, :, 1:2])(board_input)

# convolutional layers
conv1_black = tf.keras.layers.Conv2D(32, kernel_size=(2, 2), activation='relu', kernel_initializer='random_uniform', bias_initializer='zeros')(black_board)
conv2_black = tf.keras.layers.Conv2D(64, kernel_size=(2, 2), activation='relu', kernel_initializer='random_uniform', bias_initializer='zeros')(conv1_black)
flatten_black = tf.keras.layers.Flatten()(conv2_black)

conv1_red = tf.keras.layers.Conv2D(32, kernel_size=(2, 2), activation='relu',kernel_initializer='random_uniform', bias_initializer='zeros')(red_board)
conv2_red = tf.keras.layers.Conv2D(64, kernel_size=(2, 2), activation='relu',kernel_initializer='random_uniform', bias_initializer='zeros')(conv1_red)
flatten_red = tf.keras.layers.Flatten()(conv2_red)

# concatenate the flattened convolution outputs
concat = tf.keras.layers.Concatenate()([flatten_black, flatten_red, player_input])

# dense layers
dense1 = tf.keras.layers.Dense(128, activation='relu',kernel_initializer='random_uniform', bias_initializer='zeros')(concat)
dense2 = tf.keras.layers.Dense(64, activation='relu',kernel_initializer='random_uniform', bias_initializer='zeros')(dense1)

# output layer with softmax activation
output = tf.keras.layers.Dense(9, activation='softmax',kernel_initializer='random_uniform', bias_initializer='zeros')(dense2)

# define the model with board and player inputs and output layer
model = tf.keras.models.Model(inputs=[board_input, player_input], outputs=output)

# compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# Example input
#input_board = np.array([[1, 0, 2], [0, 1, 0], [2, 0, 0]])
#player = 1

# Convert to 3D tensor
#board = np.zeros((3, 3, 2))
#board[:, :, 0] = (input_board == 1)
#board[:, :, 1] = (input_board == 2)

# Add player dimension
#So, the resulting shape of the board array after this operation becomes (1, 3, 3, 2). 
# This is required because the Keras model expects the input data to be in the form of batches, 
# and the first dimension represents the batch size.
#board = np.expand_dims(board, axis=0)
#player_array = np.array([player])
#player_array = np.expand_dims(player_array, axis=0)
# print predictions
#boards = [board, board]
#players = [player_array]
#prediction = model.predict([[board, player], [board, player]])
#print(prediction)
input_boards = [np.array([[1, 0, 2], [0, 1, 0], [2, 0, 0]]),
                np.array([[2, 0, 1], [1, 2, 0], [0, 0, 0]]),
                np.array([[0, 0, 1], [0, 0, 0], [0, 0, 0]])]
input_players = [1, 2, 1]

# Convert to 4D tensor
#The input to this function in the code is board == 1 and board == 2. These are two boolean arrays, where board == 1 is 
# an array of True and False values, where True corresponds to the elements of board that are equal to 1, and False otherwise. 
# Similarly, board == 2 is an array of True and False values, where True corresponds to the elements 
# of board that are equal to 2, and False otherwise.
boards = np.stack([np.dstack((board == 1, board == 2)) for board in input_boards], axis=0)

# Add player dimension
players = np.expand_dims(input_players, axis=1)

# Make predictions
model2 = build_conv_model(board_size=3)
predictions = model2.predict([boards, players])
#print(predictions)


#state = HexStateManager.generate_initial_state(3,2)
#state_1D = state.to_1D()


#print(state_1D)
x_valid = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [2, 0, 0, 0, 0, 0, 0, 0, 1, 0],
       [1, 0, 2, 0, 0, 0, 0, 0, 1, 0],
       [2, 0, 2, 1, 0, 0, 0, 0, 1, 0],
       [1, 0, 2, 1, 0, 2, 0, 0, 1, 0]])

input_players = x_valid[:,0]
input_boards = x_valid[:, 1:].reshape(-1, 3, 3)

boards = np.stack([np.dstack((board == 1, board == 2)) for board in input_boards], axis=0)
players = np.expand_dims(input_players, axis=1)

predictions = model2.predict([boards, players])
#print(predictions)
pipeline = build_conv_pipeline(board_size=3)

predition = pipeline.predict(x_valid)
print(predition)
