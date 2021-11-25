import pandas as pd
from river import stream,tree,metrics
import utils
from encoding import prefix_bin
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
import os
from tqdm import tqdm
import sliding_window
import datetime, time
import importlib
importlib.reload(sliding_window)

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader,Dataset
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

    window_size = 100
    retraining_size = 20

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
        
    class AE_dropout(nn.Module):
        def __init__(self, input_x):
            
            super(AE_dropout, self).__init__()
            self.shape = input_x.shape
            
            hidden_factor_size = 0.2
            self.h_dim = int(self.shape[0]*self.shape[1] *hidden_factor_size)
            self.z_dim = int(self.h_dim * hidden_factor_size)
            
            if self.h_dim ==0:
                self.h_dim =1
            if self.z_dim ==0:
                self.z_dim =1
            self.fc1 = nn.Linear(self.shape[0]*self.shape[1], self.h_dim) #encode
            self.fc2 = nn.Linear(self.h_dim, self.z_dim) #encode
            
            self.fc3 = nn.Linear(self.z_dim, self.h_dim) #decode
            self.fc4 = nn.Linear(self.h_dim, self.shape[0]*self.shape[1]) #decode

            self.dout = nn.Dropout(p=0.2)
            self.relu = nn.ReLU()
            self.sigmoid = nn.Sigmoid()
            

        def encode(self, input_x):
            # x --> dropout --> fc1 --> tanh ---> dropout --> fc2 --> z
            dx = self.dout(input_x)
            h = self.relu(self.fc1(input_x))
            h = self.dout(h)
            z = self.fc2(h)
            return z

        def decode(self, z, input_x):
            # z --> dropout --> fc3 --> tanh --> dropout --> fc4 --> sigmoid --> x'
            dz = self.dout(z)
            h = self.relu(self.fc3(z))
            h = self.dout(h)
            recon_x = self.sigmoid(self.fc4(h))
            return recon_x.view(input_x.size())
        
        def forward(self, input_x):
            #flatten input and pass to encode
            z = self.encode(input_x.view(-1, self.shape[0]*self.shape[1]))
            return self.decode(z, input_x)

    def min_max_scaler(df):
        columns = df.columns.values
        
        for c in columns:
            if len(set(list(df[c]))) != 1:
                max_c = max(list(df[c]))
                min_c = min(list(df[c]))
                normalized_colum = []
                for i in list(df[c]):
                    normalized_i = (i-min_c)/(max_c - min_c)
                    normalized_colum.append(normalized_i)
                    
                df[c] = normalized_colum
        return df

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
        criterion = torch.nn.MSELoss()
        for x in pw_window:
            input_x = pw_window[x][0]
            input_x = min_max_scaler(input_x)
            dataset = pw_window[x]
            train_x,_ = Customdataset(dataset).preprocessing()
            loss_list =[]
            x_tensor = train_x[0]
            model = AE_dropout(x_tensor).cuda()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            optimizer.zero_grad()
            previous_model =0
            for i in range(10):
                running_loss =0
                for pos, x2 in enumerate(train_x):
                    
                    x_tensor = x2.cuda()
                    y_tensor = x2.cuda()
                    output = model(x_tensor)
                    loss = criterion(output, y_tensor)
                    running_loss += loss.item()
                    loss.backward()
                    optimizer.step()

    #             if len(loss_list) ==0:
    #                 pass

    #             else:
    #                 if running_loss > np.mean(loss_list):
    #                     break

                loss_list.append(running_loss)
                previous_model = model
                
            if 'detector_%s'%(x) not in training_models:
                training_models['detector_%s'%(x)] =[0,0]
            training_models['detector_%s'%(x)][0] += 1
            training_models['detector_%s'%(x)][1] = previous_model
            
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
        current_event_columns = current_event.columns.values
        c_event = Customdataset(current_event).test_preprocessing().cuda()
        
        with torch.no_grad():
            model = training_models['detector_window_%s'%(last_event.prefix_length)][1]
            test_output = model(c_event)
            output_df = pd.DataFrame(columns = current_event_columns)
            output_df.loc[0,:] = test_output.cpu().data
            
            actidxlist = []
            actlist = []
            for x in range(len(current_event_columns)):
                if 'activity_%s'%(str(last_event.prefix_length)) in current_event_columns[x]:
                    actidxlist.append(x)
                    actlist.append(current_event_columns[x])
            
            prediction = [output_df.iloc[0, actidxlist].tolist(), actlist]
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

    training_time = []

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
                train_start = time.time()
                training_models = training_stage(training_window, training_models)
                train_end = time.time()
                training_time.append(train_end-train_start)
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

    original_df = pd.read_csv(file_name)

    counting_normal = 0

    for threshold in [0.05,0.1,0.15,0.2,0.25]:
        global_true =[]
        global_pred = []

        for caseid in list(resultdict.keys()):

            prediction_list = []

            df = original_df[original_df['Case ID'] == int(caseid)].reset_index(drop=True)
            for pos, t in enumerate(resultdict['%s'%(caseid)]):
                if pos ==0:
                    continue
                
                true_act = t.event['activity']
                prediction_label = 'Normal'
                predictions = list(t.predicted.values())[0][0]
                if predictions  == 'Not Available':
                    prediction_label = 'Not Available'
                else:
                    predictions_proba = predictions[0]
                    predictions_value = predictions[1]
                    m_predictions_value = []
                    
                    for k in range(len(predictions_value)):
                        m_predictions_value.append(predictions_value[k].split('activity_%s '%(pos+1))[1])
                    
                    if true_act in m_predictions_value:
                        labelidx = m_predictions_value.index(true_act)

                        if predictions_proba[labelidx] <threshold:
                            prediction_label = 'Anomalous'
                    else:
                        prediction_label = 'Anomalous'

                if true_act != 'End':
                    prediction_list.append(prediction_label)

    #             if prediction_label == 'Anomalous':
    #                 print(true_act, m_predictions_value)
                
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

        with open('./result/ae_thr%s_window%s_retraining_%s_%s.pkl'%(threshold, window_size, retraining_size, saving_file_name), 'wb') as fp:
            pickle.dump(saving_data, fp)
