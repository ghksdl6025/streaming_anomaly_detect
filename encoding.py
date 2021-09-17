'''
Define class prefix bin to manage subperiod of case by prefix length

Information in:
1. prefix length
2. caseid
3. encoding type (ex) Aggregation, Index-base)
4. start timestamp for training window modification
5. Current event info
6. The encoded case
'''

import datetime
import utils

class prefix_bin:
    def __init__(self,caseid, event,prev_enc=None):
        self.prefix_length =0
        self.caseid = caseid
        self.event = event # Newly added event
        self.enctype = 'Aggregation'
        self.start_ts = ''
        self.prev_enc = None
        self.encoded = {} # The encoded case and event attrs
        self.predicted ={} # Historical predicted values stored
        self.true_label=None
        self.grace_updated = False

        self.set_start_ts(event['ts'])
        
    
    def _update_caseid(self, caseid):
        self.caseid = caseid
    
    def set_prefix_length(self, prefix_length):
        self.prefix_length = prefix_length
    
    def set_prev_enc(self, prev_enc):
        self.prev_enc = prev_enc
        if self.prev_enc is not None:
            self.start_ts = self.prev_enc.start_ts

    def set_start_ts(self, start_ts, format=None):
        '''
        Set start timestamp of the case for training window control
        '''
        if isinstance(start_ts, datetime.datetime):
            self.start_ts = start_ts
        else:
            self.start_ts = datetime.datetime.strptime(start_ts, '%Y-%m-%d %H:%M:%S')
    
        self.event['ts'] = datetime.datetime.strptime(start_ts, '%Y-%m-%d %H:%M:%S')

    def _set_enctype(self, enctype='Aggregation'):
        '''
        Set encoding type of the case

        Default is 'aggregation' encoding

        Parameters
        ----------
        enctype: str
            Which encoding to apply
            1) Aggregation
            2) Index-base
        '''
        self.enctype = enctype
    
    def put_event(self, event):
        '''
        Save newly added event

        Parameters
        ----------
        event: dictionary
            Newly added event
        '''
        self.event = event

    def update_encoded(self,catattrs=['activity','resource'],enctype='Aggregation'):
        '''
        Assign encoded case and event information to attrs property

        Parameters
        ----------
        event:      dictionary
            New event stream 
        
        catattrs:   list
            Categorical attributes name list

        enctype:    str
            Encoding type to apply, Aggregation or Index-base

        '''
        self._set_enctype(enctype)
        # catattrs= ['activity','resource']

        if self.enctype =='Aggregation':
            self.encoded = utils.succ_aggr_enc(self.event, catattrs= catattrs,prefix_length=self.prefix_length,prev_enc = self.prev_enc)
        
        elif self.enctype =='Index-base':
            self.encoded = utils.succ_index_enc(self.event, catattrs= catattrs,prefix_length=self.prefix_length,prev_enc = self.prev_enc)
        else:
            raise ValueError("Encoding type must be 'Aggregation' or 'Index-base'")

    def update_prediction(self, pred):
        '''
        Append newly predicted value in the self.predicted dictionary with model id as key and prediction as value

        Parameters
        ----------
        pred: value 
        '''
        # self.predicted[pred[0]] = pred[1]
        self.predicted[pred[0]]=pred[1]

    def update_truelabel(self,label):
        self.true_label = label

    def update_grace_status(self,boolean):
        self.grace_updated = boolean

    def call_predicted(self):
        return self.predicted
