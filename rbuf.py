import numpy as np

class RBUF:
    """
    class for preparing and storing the training data for ANET.
    """
    def __init__(self, board_size) -> None:
        self.X = list()
        self.y = list()
        self.board_size = board_size

    def clear(self):
        self.X = list()
        self.y = list()

    def add_case(self, state_1D, all_moves, D):
        """
        transforms the input information into the distribution over all the moves and then
        adds training case to the buffer
        """
        max_val = np.max(list(D.values()))
        for k in D:
            D[k] = D[k]/ max_val
        x = state_1D 
        all_y_valls = np.array(list(D.values()))
        #all_y_valls = all_y_valls-max(all_y_valls) # normalize so not to overflow exp
        soft_sum = np.sum(np.exp(all_y_valls))  #TODO: overloading is here! check and fix!
        y = np.array([np.e**D[move]/soft_sum if move in D  else 0 for move in all_moves])

         # check if the state already exists in the buffer
        if len(self.X)>0 and (x == self.X).all(-1).any(): 
            idx = np.where(np.all(self.X == x, axis=1))[0][0]
            self.y[idx] = y
        else:
            self.X.append(x)
            self.y.append(y)
        print(f"Number of unique elements in buffer is {len(np.unique(self.X, axis=0))}\nnumber of elements is {len(self.X)}")

    def get_training_data(self):
        return np.array(self.X), np.array(self.y,dtype='float32')
        
    def move_to_input(self, move):
        """
        Connects the move e.g. (1,2) to the corresponding index in ANET output layer, e.g. 
        """
        i,j = move
        return self.board_size*i+j
    
    def save(self, path):
        np.save(f'{path}/X.npy', self.X)    
        np.save(f'{path}/y.npy', self.y) 

    @classmethod #TODO: check
    def load(cls, path, board_size):
        rbuf = cls(board_size)
        rbuf.X = [np.array(x) for x in np.load(f'{path}/X.npy', allow_pickle=True)]
        rbuf.y = [np.array(y) for y in np.load(f'{path}/y.npy', allow_pickle=True)]
        return rbuf
        