'''

Define sliding window for streaming event prediction
1) Manage sliding window in condition


Information in:
'''
from collections import deque
import pandas as pd

class training_window:
    def __init__(self, window_size, retraining):
        self.container = deque()
        self.window_size = window_size
        self.retraining = retraining
        self.retraining_count = 0
        

        if retraining > window_size:
            raise ValueError

    def reset_retraining_count(self):
        if self.retraining_count == self.retraining:
            self.retraining_count =0

    def update_window(self, new_case):
        '''
        Add new case into the window and pull out the oldest if window size is over the condition

        Parameters
        ----------
        new_case: dict
            key: Case ID
            value: Encoded events
        '''
        self.container.append(new_case)
        self.retraining_count +=1

        if len(self.container) > self.window_size:
            self.container.popleft()

    def getAllitems(self):
        '''
        Get all items in window for training
        '''

        events = []
        for t in list(self.container):
            cid = list(t.keys())[0]
            events.append(t[cid])

        return events

    def prefix_wise_window(self):
        '''
        Construct a number of prefix-wise windows. Each window contains same length of events. 
        ----------
        Return
        prefix_window: dict
            key: Window by prefix, 'window_1','window_5'
            value: tuple with x_train and y_train
        '''

        original_window = self.getAllitems()
        max_case_length = max([len(x) for x in original_window])

        prefix_window = {}
        for t in range(max_case_length):
            window_id = 'window_%s'%(t+1)
            x_train = []
            y_train = []
            for c in original_window:
                if t+1 <=len(c):
                    x_train.append(c[t].encoded)
                    y_train.append(c[t].true_label)
            x_train = pd.DataFrame.from_dict(x_train).fillna(0)
            prefix_window[window_id] = (x_train,y_train)

        return prefix_window                

