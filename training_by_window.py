'''

Define training mechanism based on the given window
1) Prefix wise training


Information in:
'''

class training_window:
    def __init__(self, sliding_window):
        self.training_window = sliding_window

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
