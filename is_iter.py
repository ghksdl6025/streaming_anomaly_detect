import pandas as pd
from river import stream
import utils
from encoding import prefix_bin
import numpy as np
from sklearn.metrics import classification_report, accuracy_score

from tqdm import tqdm
import sliding_window
from sklearn.ensemble import IsolationForest

import datetime, time
import importlib

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
importlib.reload(sliding_window)

for contamination in [0.01]:

    for file_name in [
        './data/loan_baseline.pnml_noise_0.15_iteration_1_seed_614_sample.csv',
        './data/loan_baseline.pnml_noise_0.125_iteration_1_seed_27126_sample.csv',
        './data/loan_baseline.pnml_noise_0.09999999999999999_iteration_1_seed_14329_sample.csv',
        './data/loan_baseline.pnml_noise_0.075_iteration_1_seed_73753_sample.csv',
        './data/loan_baseline.pnml_noise_0.049999999999999996_iteration_1_seed_42477_sample.csv',
        './data/loan_baseline.pnml_noise_0.024999999999999998_iteration_1_seed_68964_sample.csv']:

        window_size = 50
        retraining_size = 1

        dataset = stream.iter_csv(
                    file_name
        #             './data/loan_baseline.pnml_noise_0.15_iteration_1_seed_614_simple.csv',
                    )

        totallength = len(list(dataset))

        dataset = stream.iter_csv(
                    file_name,
                    drop=['noise', 'lifecycle:transition', 'Variant', 'Variant index'],
                    )
        enctype = 'Index-base'

        key_pair = {
        'Case ID':'caseid',
        'Activity':'activity',
        # 'Resource':'resource',
        'Complete Timestamp':'ts',
        }
        catatars= ['activity']#,'resource']

        case_dict ={}
        training_models ={}

        casecount = 0
        rowcounter = 0
        resultdict ={}
        acc_dict ={}
        prefix_wise_window = {}
        prediction_result = {}
        graceperiod_finish=0
        finishedcases = set()

        # Sliding window for training setting

        training_window = sliding_window.training_window(window_size,retraining_size)

        def display_progress(row_counting, total_length, interval=500):
            if rowcounter%interval == 0:
                print(round(rowcounter*100/totallength,2) ,'%', 'Case finished: %s'%(casecount), 'Running cases: %s'%(len(case_dict)))

        def training_stage(window, training_models):
            '''
            Manage training stage of streaming anomaly detection
            ----------
            Parameters
            window: class training_window
                Sliding window with training data
            training_models: dict
                Trained detector by prefix stored in. Default is randomforest
            ----------
            Return
            training_models
            '''
            pw_window = window.prefix_wise_window()
            for x in pw_window:
                clf  = IsolationForest(max_samples='auto', contamination=contamination)
                
                clf.fit(pw_window[x][0])
                if 'detector_%s'%(x) not in training_models:
                    training_models['detector_%s'%(x)] =[0,0]
                training_models['detector_%s'%(x)][0] += 1
                training_models['detector_%s'%(x)][1] = clf
            return training_models

        def predict_activity_proba(last_event):
            '''
            Predict next activity prediction 
            
            Parameters
            ----------
            last_event: case_bin
            
            Return
            ----------
            modelid, prediction
            
            '''
            feature_matrix = prefix_wise_window['window_%s'%(last_event.prefix_length)][0].columns.values
            current_event = utils.readjustment_training(last_event.encoded, feature_matrix)
            current_event = pd.Series(current_event).to_frame().T
            prediction = [training_models['detector_window_%s'%(last_event.prefix_length)][1].predict_proba(current_event), training_models['detector_window_%s'%(last_event.prefix_length)][1].classes_]
            modelid = training_models['detector_window_%s'%(last_event.prefix_length)][0]

            return modelid, prediction

        def first_event(case_bin):
            '''
            Generate start event before first event
            '''
            print(case_bin.event['ts'])
            empty_data ={'activity':'Start signal', 'ts':datetime.datetime.strftime(case_bin.event['ts'], '%Y-%m-%d %H:%M:%S')}
            start_event = prefix_bin(case_bin.caseid, empty_data)
            start_event.set_prefix_length(0)
            start_event.update_encoded(catattrs=catatars,enctype=enctype)
            start_event.update_truelabel(case_bin.event['activity'])
            return start_event

        start_time = time.time()

        for x,y in dataset:
            display_progress(rowcounter, totallength)
            rowcounter +=1
            
            utils.dictkey_chg(x, key_pair)
            # Event stream change dictionary keys
            x['ts'] = x['ts'][:-4]
            
            # Check label possible
            
            # Initialize case by prefix length
            caseid = x['caseid']
            x.pop('caseid')
            
            case_bin = prefix_bin(caseid, x)
            
            if caseid not in list(case_dict.keys()):
                case_dict[caseid] = []
                case_bin.set_prefix_length(1)
                
            elif caseid in finishedcases:
                continue
            
            else:
                case_bin.set_prefix_length(len(case_dict[caseid])+1)
                case_bin.set_prev_enc(case_dict[caseid][-1])
            
            # Encode event and cases and add to DB
            ts = case_bin.event['ts']
            case_bin.update_encoded(catattrs=catatars,enctype=enctype)
            
            # Set current activity as outcome of previous event
            if case_bin.prefix_length != 1:
                case_bin.prev_enc.update_truelabel(x['activity'])

            # First prediction for current event
            
            last_event = case_bin
            modelid = 'None'
            prediction = 'Not Available'

            if len(training_window.getAllitems()) !=0:
                if 'window_%s'%(last_event.prefix_length) in list(prefix_wise_window.keys()) and 'detector_window_%s'%(last_event.prefix_length) in training_models.keys():
        #             modelid, prediction = predict_activity_proba(last_event)
                    feature_matrix = prefix_wise_window['window_%s'%(last_event.prefix_length)][0].columns.values
                    current_event = utils.readjustment_training(last_event.encoded, feature_matrix)
                    current_event = pd.Series(current_event).to_frame().T
                    prediction = [training_models['detector_window_%s'%(last_event.prefix_length)][1].predict(current_event)]
                    modelid = training_models['detector_window_%s'%(last_event.prefix_length)][0]
            case_bin.update_prediction((modelid, (prediction,ts)))        
                    
            # Update training window and finish the case
            if x['activity'] == 'End':
                training_window.update_window({caseid: case_dict[caseid]})        
                if training_window.retraining == training_window.retraining_count:            
                    training_models = training_stage(training_window, training_models)
                    prefix_wise_window = training_window.prefix_wise_window()
                    
                resultdict[caseid] = case_dict[caseid]
                case_dict.pop(caseid)

                casecount +=1
                for x in case_dict:
                    last_event = case_dict[x][-1]
                    modelid = 'None'
                    prediction = 'Not Available'

                    if len(training_window.getAllitems()) !=0:
                        prefix_wise_window = training_window.prefix_wise_window()
                        if 'window_%s'%(last_event.prefix_length) in list(prefix_wise_window.keys()) and 'detector_window_%s'%(last_event.prefix_length) in training_models.keys():
        #                     modelid, prediction = predict_activity_proba(last_event)

                            feature_matrix = prefix_wise_window['window_%s'%(last_event.prefix_length)][0].columns.values
                            current_event = utils.readjustment_training(last_event.encoded, feature_matrix)
                            current_event = pd.Series(current_event).to_frame().T
                            prediction = [training_models['detector_window_%s'%(last_event.prefix_length)][1].predict(current_event)]
                            modelid = training_models['detector_window_%s'%(last_event.prefix_length)][0]
                    case_dict[x][-1].update_prediction((modelid, (prediction,ts)))        
                training_window.reset_retraining_count()
            else:
                case_dict[caseid].append(case_bin)


        end_time = time.time()

        original_df = pd.read_csv(file_name)

        for_confusion_matrix = {}

        global_true =[]
        global_pred = []
        counting_normal = 0
        for caseid in list(resultdict.keys()):

            for_confusion_matrix[int(caseid)] =[]
            
            prediction_list = []
            
            df = original_df[original_df['Case ID'] == int(caseid)].reset_index(drop=True)
            for pos, t in enumerate(resultdict['%s'%(caseid)]):
                
                predictions = list(t.predicted.values())[0][0]    
                if predictions  == 'Not Available':
                    predictions_label = 'Not Available'
                else:
                    predictions_label = predictions[0][0]

                if predictions_label == 1:
                    predictions_label = 'Normal'
                elif predictions_label == -1:
                    predictions_label = 'Anomalous'

                if t.event['activity'] != 'Start':
                    prediction_list.append(predictions_label)
                    
            true_label_list = []

            labellist = list(df['noise'])
            actlist = list(df['Activity'])
            for pos, t in enumerate(labellist):
                if t == 'Start' or t == 'End':
                    continue
                elif t == 'true':
                    true_label = 'Anomalous'
                else:
                    true_label = 'Normal'
                true_label_list.append(true_label)

            
            for pos, p in enumerate(prediction_list):
                global_pred.append(p)
                global_true.append(true_label_list[pos])
        saving_data = {'y_true':global_true, 'y_pred':global_pred}
        import pickle
        saving_file_name = file_name.split('/')[-1][:-4]

        if retraining_size ==1:
            retraining_size =0

        with open('./result/iso_cont%s_window%s_retraining_%s_%s.pkl'%(contamination, window_size, retraining_size, saving_file_name), 'wb') as fp:
            pickle.dump(saving_data, fp)

