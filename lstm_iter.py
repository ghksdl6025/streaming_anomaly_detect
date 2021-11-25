import pandas as pd
from river import stream
import utils
from encoding import prefix_bin
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
import os
import sliding_window
import datetime, time
import importlib
importlib.reload(sliding_window)

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

import torch.nn.functional as F
import torch.optim as optim

#torch cuda setting
with torch.no_grad():
    torch.cuda.empty_cache()

device = 'cuda' if torch.cuda.is_available() else 'cpu'



for file_name in [
    './data/loan_baseline.pnml_noise_0.15_iteration_1_seed_614_sample.csv',
    './data/loan_baseline.pnml_noise_0.125_iteration_1_seed_27126_sample.csv',
    './data/loan_baseline.pnml_noise_0.09999999999999999_iteration_1_seed_14329_sample.csv',
    './data/loan_baseline.pnml_noise_0.075_iteration_1_seed_73753_sample.csv',
    './data/loan_baseline.pnml_noise_0.049999999999999996_iteration_1_seed_42477_sample.csv',
    './data/loan_baseline.pnml_noise_0.024999999999999998_iteration_1_seed_68964_sample.csv']:

    window_size = 50
    retraining_size = 25


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

    def display_progress(row_counting, total_length, start_time, interval=500):
        if rowcounter%interval == 0:
            print(round(rowcounter*100/totallength,2) ,'%', 'Case finished: %s'%(casecount), 'Running cases: %s'%(len(case_dict)),
                'Elapse time: %s mins'%(round((time.time()-start_time)/60, 3)))

    class Customdataset():
        def __init__(self, dataset):
            '''
            Convert dataset to tensor
            
            Params
            dataset_type: Type of dataset, trainset, validset, and testset
            '''
            self.dataset = dataset


        def preprocessing(self):
            self.x_data=self.dataset[0]
            self.y_data=self.dataset[1]

            x = self.x_data.to_numpy()
            x = np.reshape(x, (x.shape[0],1, x.shape[1]))
            y_set = sorted(set(self.y_data))
            train_y =[]
            for y in self.y_data:
                train_y.append(y_set.index(y))

            x_tensor = torch.tensor(x, dtype=torch.float)
            y_tensor = torch.tensor(train_y, dtype=torch.long)

            return x_tensor, y_tensor

        
        def test_preprocessing(self):
            self.x_data=self.dataset

            x = self.x_data.to_numpy()
            x = np.reshape(x, (x.shape[0],1, x.shape[1]))

            x_tensor = torch.tensor(x, dtype=torch.float)

            return x_tensor
        

    class LSTM_model(nn.Module): # nn.Module inherit

        def __init__(self, input_x, raw_y):
            super(LSTM_model, self).__init__()
            
            self.input_size = input_x.shape[2]
            self.hidden_size =2* input_x.shape[2]
            self.num_case = 1
            self.num_layers =2

            self.lstm = nn.LSTM(self.input_size, self.hidden_size, num_layers=self.num_layers, dropout=0.25, batch_first =False, bidirectional = False)

            self.h0 = torch.randn(self.num_layers, 1, self.hidden_size, device=device)
            self.c0 = torch.randn(self.num_layers, 1, self.hidden_size, device=device)

            latent_vector_size =50 * 1
            self.linear1 = nn.Linear(1 *self.num_case *self.hidden_size, latent_vector_size)
            self.linear_h = nn.Linear(1 *self.num_layers *self.hidden_size, latent_vector_size)
            self.linear_o = nn.Linear(3 * latent_vector_size, 1 *self.num_case * len(set(raw_y)))

            self.relu = nn.ReLU()


        def forward(self, input_x):
            output, (hn,cn) = self.lstm(input_x, (self.h0,self.c0))
            output = output.reshape((output.size()[0] *output.size()[1] *output.size()[2]))
            output = self.relu(self.linear1(output))

            uH = F.leaky_relu(self.linear_h(hn.reshape((hn.size()[0] *hn.size()[1] *hn.size()[2]))))
            uC = F.leaky_relu(self.linear_h(cn.reshape((cn.size()[0] *cn.size()[1] *cn.size()[2]))))
            output = torch.cat((uH ,uC ,output))
            output = self.linear_o(output)
            
            output =output.reshape(self.num_case,-1)

            return output
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
        pw_window = training_window.prefix_wise_window()
        cross_entropy_loss = torch.nn.CrossEntropyLoss()
        for x in pw_window:
            input_x = pw_window[x][0]
            input_y = pw_window[x][1]
            dataset = pw_window[x]
            train_x,train_y = Customdataset(dataset).preprocessing()
            loss_list =[]

            x_tensor = torch.unsqueeze(train_x[0], dim=0)
            y_tensor = torch.unsqueeze(train_y[0], dim=0)
            model = LSTM_model(x_tensor, input_y).cuda()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            optimizer.zero_grad()

            previous_model =0
            for i in range(10):
                running_loss =0
                for pos, x2 in enumerate(train_x):
                    x_tensor = torch.unsqueeze(x2, dim=0).cuda()
                    y_tensor = torch.unsqueeze(train_y[pos], dim=0).cuda()

                    output = model(x_tensor)
                    loss = cross_entropy_loss(output, y_tensor)
                    running_loss += loss.item()
                    loss.backward()
                    optimizer.step()

                if len(loss_list) ==0:
                    pass

                else:
                    if running_loss > np.mean(loss_list):
                        break
                    
                loss_list.append(running_loss)
                previous_model = model
                
            if 'detector_%s'%(x) not in training_models:
                training_models['detector_%s'%(x)] =[0,0,0]
            training_models['detector_%s'%(x)][0] += 1
            training_models['detector_%s'%(x)][1] = previous_model
            training_models['detector_%s'%(x)][2] = sorted(set(input_y))
            
            del x_tensor
            del y_tensor
            del previous_model
            del model
            torch.cuda.empty_cache()
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
        current_event = Customdataset(current_event).test_preprocessing().cuda()

        with torch.no_grad():
            model = training_models['detector_window_%s'%(last_event.prefix_length)][1]
            label_classes = training_models['detector_window_%s'%(last_event.prefix_length)][2]
            test_output = model(current_event)
            prediction = [test_output, label_classes]
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
        display_progress(rowcounter, totallength, start_time)
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
                modelid, prediction = predict_activity_proba(last_event)
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
                        modelid, prediction = predict_activity_proba(last_event)

                case_dict[x][-1].update_prediction((modelid, (prediction,ts)))        
            training_window.reset_retraining_count()
        else:
            case_dict[caseid].append(case_bin)


    end_time = time.time()
    print(round((end_time - start_time)/60,3))
    original_df = pd.read_csv(file_name)

    for_confusion_matrix = {}

    counting_normal = 0

    for threshold in [0.01]:
        global_true =[]
        global_pred = []

        for caseid in list(resultdict.keys()):

            for_confusion_matrix[int(caseid)] =[]

            prediction_list = []

            df = original_df[original_df['Case ID'] == int(caseid)].reset_index(drop=True)
            for pos, t in enumerate(resultdict['%s'%(caseid)]):
                prediction_label = 'Normal'
                predictions = list(t.predicted.values())[0][0]
                if predictions  == 'Not Available':
                    prediction_label = 'Not Available'
                else:
                    predictions_proba = F.softmax(predictions[0][0])
                    predictions_value = list(predictions[1])

                    if t.true_label in predictions_value:
                        labelidx = predictions_value.index(t.true_label)

                        if predictions_proba[labelidx] <threshold:
                            prediction_label = 'Anomalous'
                    else:
                        prediction_label = 'Anomalous'

                if t.true_label != 'End':
                    prediction_list.append(prediction_label)



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

        with open('./result/lstm_thr%s_window%s_retraining_%s_%s.pkl'%(threshold, window_size, retraining_size, saving_file_name), 'wb') as fp:
            pickle.dump(saving_data, fp)
