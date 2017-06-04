from chessboard import chessboard
from itertools import combinations_with_replacement as comb
import numpy as np
import scipy
from random import choice
from scipy.sparse import coo_matrix as sparse
from scipy.io import mmwrite, mmread

nine = np.matrix([[1,2,3],[4,5,6],[7,8,9]])

class Qtic():
    def __init__(self, gamma=0.5, board_size=3, win=3):
        self.gamma = gamma
        self.chessboard = chessboard.Chessboard(board_size=board_size, win=win)
        self.Q_file = 'Q.mtx'
        self.index = 1
        self.next_index = 1
        try:
            self.Q = mmread(self.Q_file)
        except:
            self.Q = sparse((3**board_size, board_size**2))

    def get_actions(self):
        actions = []
        for i, j in comb(self.chessboard.pos_range, 2):
            if self.chessboard.pos[i][j] == 0:
                actions.append((i, j))
        return actions

    def reward(self, action, run):
        if self.chessboard.check_win(action[0], action[1], run):
            return 50
        return 0

    def save_q(self):
        mmwrite(self.Q_file, self.Q)

    def train(self):
        run = 1
        game_round = 1
        self.chessboard.print_pos()
        while True:
            if (run%2 == 1):
                #player's turn
                inp = input('Input pos:')
                inp = inp.replace(' ', '')
                y, x = inp.split(',')
                x, y = int(x) - 1, int(y) - 1
                self.index *= nine[x, y]
                result = self.chessboard.set_pos(x, y, run)
                self.chessboard.print_pos()
                if result:
                    print('Player 1 wins')
                    return
            else:
                actions = self.get_actions()
                for action in actions:
                    c_reward = self.reward(action, run)
                    c_ind = nine[action] - 1
                    self.next_index = self.index*nine[action] - 1
                    self.Q[self.index, c_ind] = c_reward + self.gamma*max(self.Q[self.next_index])

                result = self.chessboard.set_pos(x, y, run)
                self.chessboard.print_pos()
                if result:
                    print('Player 2 wins')
                    return
            run = 2 - (run + 1)%2
            game_round += 1
            if game_round == 10:
                print('DRAW!')
                return

q = Qtic()
q.train()
q.save_q()

