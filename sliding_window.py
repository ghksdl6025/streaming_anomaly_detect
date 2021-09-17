'''

Define sliding window for streaming event prediction
1) Manage sliding window in condition


Information in:
'''
from collections import deque

class training_window:
    def __init__(self, window_size):
        self.container = deque()
        self.window_size = window_size

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

        if len(self.container) > self.window_size:
            self.container.popleft()

    def getAllitems(self):
        '''
        Get all items in window for training
        '''
        return list(self.container)

