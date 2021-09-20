'''
Utility functions used in training and pre-processing stage.
'''
import graphviz 
from collections import Counter
import datetime

def save_graph_as_png(dot_string, output_file):
    if type(dot_string) is str:
        g = graphviz.Source(dot_string)
    elif isinstance(dot_string, (graphviz.dot.Digraph, graphviz.dot.Graph)):
        g = dot_string
    if '/' in output_file:
        dir_add = output_file.split('/')[1]
        output_file = output_file.split('/')[2]
    g.format = 'png'
    g.filename = output_file
    g.directory = './img/' +dir_add
    g.render(view=False,cleanup=True)

    return g

def dictkey_chg(dictionary, key_pair):
    '''
    Change dictionary key from old to new according to key_pair

    Parameters
    ----------
    dictionary : Dictionary
        dictionary to modify

    key_pair : Dictionary
        key pair with old (key) and new (value)

    Return
    ----------
    Modified dictionary
    '''
    for old_key in list(key_pair.keys()):
        dictionary[key_pair[old_key]] = dictionary.pop(old_key)
    return dictionary

def set_label(data):
    '''
    From given data, set case label if condition satisfied.
    Return data with case label y.

    Parameters
    ----------
    data : dictionary
    Streaming data

    Return
    ----------
    data : updated dictionary
    '''
    y = ''
    if 'Cancelled' in data['activity']:
        y = 'Not accepted'
    elif 'Accepted' in data['activity']:
        y = 'Accepted'
    elif 'Refused' in data['activity']:
        y = 'Not accepted'

    if len(y) >=1:
        data['True label'] =y
    return data

def succ_aggr_enc(event,catattrs,prefix_length,prev_enc=None):
    '''
    Succeding aggregation encoding
    Aggregation encoding method along previously encoded case information in aggregation method

    Parameters
    ----------
    event: dictionary
        Newly added event
    
    catattrs: list
        List of categorical attributes

    prefix_length: int
        Current case prefix length

    prev_enc: class
        Previously streamed event case


    Returns
    ----------
    One hot encoded dictionary in aggregation based 
    '''
    if prev_enc == None:
        ohedict ={}
    else:
        ohedict = {x:prev_enc.encoded[x] for x in prev_enc.encoded.keys()}

    # Event duration average
    if 'duration_avg' not in list(ohedict.keys()):
        ohedict['duration_avg'] = 0
    else:
        # Current event duration
        duration = (event['ts'] - prev_enc.event['ts']).total_seconds()

        # Calculate average event duration | ((prev_avg)*(prefixlength -1) + duration)/prefixlength
        ohedict['duration_avg'] = (ohedict['duration_avg']*(prefix_length-1) + duration)/prefix_length

    # categorical attributes one hot encoding and counting
    for cat in catattrs:
        oheitem = cat+' '+event[cat]
        if oheitem not in list(ohedict.keys()):
            ohedict[oheitem] = 1
        else:
            ohedict[oheitem] +=1
        
    return ohedict

def succ_index_enc(event,catattrs,prefix_length,prev_enc=None):
    '''
    Succeding index-base encoding
    Index-base encoding method along previously encoded case information in same method

    Parameters
    ----------
    event: dictionary
        Newly added event
    
    catattrs: list
        List of categorical attributes

    prefix_length: int
        Current case prefix length

    prev_enc: class
        Previously streamed event case


    Returns
    ----------
    One hot encoded dictionary in aggregation based 
    '''
    if prev_enc == None:
        ohedict ={}
    else:
        ohedict = {x:prev_enc.encoded[x] for x in prev_enc.encoded.keys()}

    # Event duration and cumlative duration
    if 'duration_%s'%(prefix_length-1) not in list(ohedict.keys()):
        ohedict['duration_%s'%(prefix_length)] = 0
        ohedict['cumduration_%s'%(prefix_length)] = 0
    else:
        # Current event duration in seconds
        duration = (event['ts'] - prev_enc.event['ts']).total_seconds()
        ohedict['duration_%s'%(prefix_length)] =  duration
        ohedict['cumduration_%s'%(prefix_length)] = ohedict['cumduration_%s'%(prefix_length-1)] + duration

    # categorical attributes one hot encoding and counting
    for cat in catattrs:
        oheitem = cat+'_%s '%(prefix_length)+event[cat]
        ohedict[oheitem] = 1
        
    return ohedict


def invoke_cases_by_prefix(case_dict):
    '''
    Invoke encoded cases by given prefix length and return 

    Parameters
    ----------
    case_dict: Dictionary
        Dictionary with cases contains 'bin' class 
    
    prefix: int
        Prefix length to gather
    
    Return
    ----------
    Gathered encoded cases in dictionary
    '''
    for caseid in list(case_dict.keys()):
        for prefix in range(1,len(case_dict[caseid])):
            casebyprefix = case_dict[caseid][prefix]
            print(casebyprefix.prefix_length, casebyprefix.encoded)

def readjustment_training(dataset, feature_matrix):
    '''
    Readjustment on encoded dataset and extract feature matrix

    Parameters
    ----------
    dataset: dictionary
        Dictionary of encoded event

    feature_matrix: list
        List of feature matrix for specific prefix length

    Return
    ----------
    Encoded event with updating features from feature matrix
    '''
    new_dataset ={}
    for feature in feature_matrix:
        if feature in list(dataset.keys()):
            new_dataset[feature] = dataset[feature]
        else:
            new_dataset[feature] = 0
    return new_dataset


'''
Functions for continuous evaluation
1) averaged_prediction
2) get_bin-list
3) averaged_prediction_by_bin
4) rt_event_continuous_evaluation (relative time-event)

'''

def averaged_prediction(bin_result_list):
    '''
    Get single prediction value by bin result list

    Parameters
    ----------
    bin_result_list: list
        List of predicted values of the bin
    
    Return
    ----------
    The most frequent singe value in arg max value from bin_result_list
    '''
    try:
        answer =sorted(Counter(bin_result_list).items(),key = (lambda x:x[1]),reverse=True)[0][0] 
    except:
        answer = None
    return answer


def get_ts_bin_list(current_ts, next_ts, bin_n):
    '''
    Return binlist from start timestamp to the next/end timestamp divided by number of bins

    Parameters
    ----------
    current_ts:
        Start timestamp

    next_ts:
        Next or end timestamp
    
    bin_n:
        Number of bins to divide

    Return
    ----------
        Bin list contains 2-value tuple with start and end of each bin ex) [(b1_start, b1_end), (b2_start, b2_end)...] #b1_end = b2_start
    '''
    binsize = (next_ts -current_ts).total_seconds()/bin_n
    binlist= []
    prev_bin=current_ts
    for t in range(bin_n):
        binlist.append((prev_bin, prev_bin +datetime.timedelta(seconds=binsize)))
        prev_bin = prev_bin +datetime.timedelta(seconds=binsize)

    return binlist

def get_pl_bin_list(pl_start, pl_end):
    '''
    Return binlist from start prefix length to the next/end prefix length by each event
    ex) [(e2,e3),(e3,e4)....]

    Parameters
    ----------
    pl_start:
        Start prefix length

    pl_end:
        Next or end prefix length
    
    Return
    ----------
        Bin list contains 2-value tuples with start and end of each bin 
    '''
    binsize = pl_end -pl_start
    binlist= []
    prev_bin=pl_start
    for t in range(binsize):
        binlist.append((prev_bin, prev_bin +1))
        prev_bin = prev_bin +1

    return binlist


def ts_averaged_prediction_by_bin(bin_list, event_prediction):
    bin_result_dict = {}
    for pos, t in enumerate(bin_list):
        t = (t[0],t[1],pos+1)
        bin_result_dict[t] = []
    prev_bin_t = list(bin_result_dict.keys())[0]
    for bin_t in bin_result_dict.keys():
        for result in event_prediction.values():
            if result[1] >=  bin_t[0] and result[1] <  bin_t[1]:
                bin_result_dict[bin_t].append(result[0])
            elif bin_t[0]==bin_t[1] and result[1] == bin_t[0]:
                bin_result_dict[bin_t].append(result[0])                       
        if len(bin_result_dict[bin_t]) ==0:
            bin_result_dict[bin_t].append(bin_result_dict[prev_bin_t])
        prev_bin_t = bin_t
        bin_result_dict[bin_t] = averaged_prediction(bin_result_dict[bin_t])
    return bin_result_dict

def pl_case_continuous_evaluation(resultdict):
    bin_pred = {}
    y_true = {}
    for case in resultdict.keys():
        if len(resultdict[case]) > 2:
            for event in range(1,len(resultdict[case])-1):
                bin_result_list = [x[0] for x in resultdict[case][event].predicted.values()]
                bin_predicted_value = averaged_prediction(bin_result_list)
                if bin_predicted_value != None:
                    y_true[str(case)+'_'+str(event+1)] = resultdict[case][event].true_label
                    bin_pred[str(case)+'_'+str(event+1)] = bin_predicted_value
    return y_true, bin_pred


def rt_event_continuous_evaluation(resultdict, bin_n):
    bin_pred = {}
    y_true = {}
    for case in resultdict.keys():
        if len(resultdict[case]) > 2:
            for event in range(1,len(resultdict[case])-1):
                current_event_ts = resultdict[case][event].event['ts']
                next_event_ts = resultdict[case][event+1].event['ts']
                bin_list = get_ts_bin_list(current_event_ts, next_event_ts, bin_n)
                try:
                    t = ts_averaged_prediction_by_bin(bin_list,resultdict[case][event].predicted)
                    y_true[str(case)+'_'+str(event+1)]=resultdict[case][event].true_label
                    bin_pred[str(case)+'_'+str(event+1)]=list(t.values())
                except:
                    pass
    
    return y_true, bin_pred

def rt_case_continuous_evaluation(resultdict, bin_n):
    bin_pred = {}
    y_true = {}
    for case in resultdict.keys():
        if len(resultdict[case]) > 2:
            prediction_counter = 0
            prediction_by_bin ={}
            start_ts = resultdict[case][1].event['ts']
            end_ts = resultdict[case][-1].event['ts']
            bin_list = get_ts_bin_list(start_ts, end_ts, bin_n)
            for event in range(1,len(resultdict[case])-1):
                for predict in resultdict[case][event].predicted.keys():
                    prediction_by_bin[prediction_counter] = resultdict[case][event].predicted[predict]
                    prediction_counter +=1
            t = ts_averaged_prediction_by_bin(bin_list,prediction_by_bin)
            for each_bin in [x for x in t.keys()]:                
                y_true[str(case)+'_'+str(each_bin[2]+1)]=resultdict[case][event].true_label
                bin_pred[str(case)+'_'+str(each_bin[2]+1)]= t[each_bin]
    return y_true, bin_pred

class window_evaluation:
    def __init__(self, window_size):
        self.acc_window_savings =[]
        self.acc_window=[]
        self.window_size = window_size
    
    def calculate_acc(self):
        counting_true = 0
        for yt, yp in self.acc_window:
            if yt ==yp:
                counting_true +=1
        return counting_true / len(self.acc_window)

    def update(self,yt,yp):
        if len(self.acc_window) >= self.window_size:
            self.acc_window.pop(0)
        self.acc_window.append((yt,yp))
        self.acc_window_savings.append(self.calculate_acc())
    
    def invoke_acc_savings(self):
        return self.acc_window_savings

    def get(self):
        if len(self.acc_window_savings) >=1:
            return self.acc_window_savings[-1]
        else:
            return 0

    def label_dist(self):
        label_list = [x[0] for x in self.acc_window]
        return Counter(label_list)
