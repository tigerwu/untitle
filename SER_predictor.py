#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys
import time
import wave
import pickle
from scipy.io import wavfile
import aubio
from aubio import source,pitch
from subprocess import call
import time
import pandas as pd
import numpy as np
from tsfresh.transformers import RelevantFeatureAugmenter
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from pyAudioAnalysis import ShortTermFeatures


# In[2]:


def Score(pred_y,p):
    if pred_y==0:
        p=sorted(p, reverse = True)
        score=-(p[0]/(p[0]+p[1]))
        score=(score+0.75)*7/5-0.65
        score=round(score,2)
    elif pred_y==1:
        p=sorted(p, reverse = True)
        score=p[0]/(p[0]+p[1])
        score=(score-0.75)*7/5+0.65
        score=round(score,2)
    elif pred_y==2 and p[0]>p[1]:
        p=sorted(p, reverse = True)
        score=-((1-(p[0]/(p[0]+p[1])))*0.6)
        score=round(score,2)
    else:
        p=sorted(p, reverse = True)
        score=(1-(p[0]/(p[0]+p[1])))*0.6
        score=round(score,2)
    return score


# In[3]:


class predictor(object):
    def __init__(self, path_to_svm='Model_ms.m',
                 path_to_scaler='Scaler_ms.m',
                 path_to_selector_f0='selector_ms_fo_0.05.m',
                 path_to_selector_short='selector_ms_py_0.05.m'):
        self.dic={0:'生气',1:'高兴',2:'平和'}
        self.path_to_svm=path_to_svm
        self.path_to_scaler=path_to_scaler
        self.path_to_selector_short=path_to_selector_short
        self.path_to_selector_f0=path_to_selector_f0
        if not self.load_svm():
            raise ValueError ('failed to load svm, please check path')
        if not self.load_scaler():
            raise ValueError ('failed to load scaler, please check path')
        if not self.load_selector_f0():
            raise ValueError ('failed to load selector_f0, please check path')
        if not self.load_selector_short():
            raise ValueError ('failed to load selector_short, please check path')
    
    def aubio_pitch(self,path):
        sr,y=self.read_wav(path)
        downsample = 1
        samplerate = sr // downsample

        win_s = 2048 // downsample # fft size
        hop_s = 80  // downsample # hop size

        s = source(path, samplerate, hop_s)
        samplerate = s.samplerate

        tolerance = 0.8

        pitch_o = aubio.pitch("mcomb", win_s, hop_s, samplerate)#####自相关法，与opensmile一致

        pitch_o.set_unit("Hz")
        pitch_o.set_tolerance(tolerance)

        pitches = []
        confidences = []

        # total number of frames read
        total_frames = 0
        while True:
            samples, read = s()
            pitch = pitch_o(samples)[0]
            #pitch = int(round(pitch))
            confidence = pitch_o.get_confidence()
            #if confidence < 0.8: pitch = 0.
            #print("%f %f %f" % (total_frames / float(samplerate), pitch, confidence))
            pitches += [pitch]
            confidences += [confidence]
            total_frames += read
            if read < hop_s: break
        #print(len(pitches))
        #print(pitches)
        return pitches
    def load_svm(self):
        try:
            with open(self.path_to_svm,'rb') as fp:
                self.clf=pickle.load(fp)
        except:
            #print('fail to load svm, examine the path to svm.') 
            return 0
        return 1
    def load_selector_f0(self):
        try:
            with open(self.path_to_selector_f0,'rb') as fp:
                self.selector_f0=pickle.load(fp)
        except:
            return 0
        return 1
    def load_selector_short(self):
        try:
            with open(self.path_to_selector_short,'rb') as fp:
                self.selector_short=pickle.load(fp)
        except:
            return 0
        return 1
    def load_scaler(self):
        try:
            with open(self.path_to_scaler,'rb') as fp:
                self.scaler=pickle.load(fp)
        except:
            return 0
        return 1
    def read_wav(self, path):
        f = wave.open(path,"rb")

        params = f.getparams()  
        nchannels, sampwidth, framerate, nframes = params[:4]
        str_data  = f.readframes(nframes)  
        f.close()
        wave_data = np.frombuffer(str_data,dtype = np.short)
        if len(wave_data) == 0:
            raise ValueError('got empty wav file:'+path)
        return framerate, wave_data    
    def get_short_features(self,file_path):
        try:
            Fs, x = self.read_wav(file_path)
        except:
            print('fail to extract short '+ file_path+',passed.')
            return pd.DataFrame()
        audio_name=file_path.split('/')[-1]
        #emotion=file_path.split('/')[-2]
        #outpath=out_path+emotion+'_'+audio_name[:-4]+'.txt'
        #[Fs, x] = audioBasicIO.read_audio_file(file_path)
        F_s,F_name=ShortTermFeatures.feature_extraction(x,Fs,0.05*Fs,0.025*Fs)
        #F_m,F_s,F_name=self.mid_feature_extraction(x,Fs,1.0*Fs,0.5*Fs,0.05*Fs,0.025*Fs)
        
        short=pd.DataFrame(F_s.T)
        short['id']=file_path.split('/')[-1]
        return short
    def get_pitch(self,file_path):
        try:
            pitches= self.aubio_pitch(file_path)
        except:
            print('fail to read '+ file_path+',passed.')
            return pd.DataFrame()
        data=pd.DataFrame(pitches)
        data['id']=file_path.split('/')[-1]
        #data['time']=pd.Series(data.index)###保持与之前代码一致，不是必须的
        #print(data)
        return data
    def single_pitchs_and_labels(self,path_to_audio):
        df_data=pd.DataFrame()

        y_label={}

        pitch=self.get_pitch(path_to_audio)
        #print(pitch)
        if len(pitch)==0:
            raise ValueError ('failed to get pitch:'+path_to_audio)
        df_data=pd.concat([df_data,pitch],ignore_index=True)
        
        y_label[pitch['id'][0]]='unknown'

        y_label=pd.Series(y_label)
        sample_names = pd.DataFrame(index=y_label.index)
        return df_data,y_label,sample_names
    def single_short_and_labels(self,path_to_audio):
        df_data=pd.DataFrame()

        y_label={}

        pitch=self.get_short_features(path_to_audio)
        if len(pitch)==0:
            raise ValueError ('failed to get features:'+path_to_audio)
        df_data=pd.concat([df_data,pitch],ignore_index=True)
        
        y_label[pitch['id'][0]]='unknown'

        y_label=pd.Series(y_label)
        sample_names = pd.DataFrame(index=y_label.index)
        return df_data,y_label,sample_names
    def batch_pitchs_and_labels(self,path_to_audios):
        df_data=pd.DataFrame()
        y_label={}
        for f in os.listdir(path_to_audios):
            if not f.endswith('wav'):#####attention the endwith
                continue

            
            pitch=self.get_pitch(path_to_audios+f)
            #print(pitch)
            if len(pitch)==0:#failed to read
                continue
            df_data=pd.concat([df_data,pitch],ignore_index=True)
            y_label[pitch['id'][0]]='unknown'
        if len(y_label)==0:
            raise ValueError ('no .wav file found in '+path_to_audios)
        y_label=pd.Series(y_label)
        sample_names = pd.DataFrame(index=y_label.index)
        return df_data,y_label,sample_names    
    def batch_short_and_labels(self,path_to_audios):
        df_data=pd.DataFrame()
        y_label={}
        for f in os.listdir(path_to_audios):
            if not f.endswith('wav'):#####attention the endwith
                continue

            
            pitch=self.get_short_features(path_to_audios+f)
            #print(pitch)
            if len(pitch)==0:#failed to read
                continue
            df_data=pd.concat([df_data,pitch],ignore_index=True)
            y_label[pitch['id'][0]]='unknown'
        if len(y_label)==0:
            raise ValueError ('no .wav file found in '+path_to_audios)
        y_label=pd.Series(y_label)
        sample_names = pd.DataFrame(index=y_label.index)
        return df_data,y_label,sample_names 
    def show_result(self,sample_names,y_hat):
        for i in range(len(sample_names)):
            print(np.array(sample_names.index)[i],self.dic[y_hat[i]])
    def single_wav_test(self,path_to_audio):
        df_data,y_label,sample_names=self.single_pitchs_and_labels(path_to_audio)
        df_data1,y_label1,sample_names1=self.single_short_and_labels(path_to_audio)
        self.selector_f0.set_timeseries_container(df_data)
        test_features=self.selector_f0.transform(sample_names)
        
        
        self.selector_short.set_timeseries_container(df_data1)
        test_features1=self.selector_short.transform(sample_names1)
        
        
        test_features=pd.merge(test_features,test_features1,how= 'inner',left_index=True,right_index=True)
        test_features = self.scaler.transform(test_features)  
        y_hat=self.clf.predict(test_features)
        y_hat_pro=np.array(self.clf.predict_proba(test_features))
        y_hat_pro=np.around(y_hat_pro, decimals=2)
        
        s=[]

        for i in range(len(y_hat_pro)):
            if y_hat_pro[i][0]>y_hat_pro[i][1] and y_hat_pro[i][0]/y_hat_pro[i][2]>1:
                y_pro=0
            elif y_hat_pro[i][1]>y_hat_pro[i][0] and y_hat_pro[i][1]>y_hat_pro[i][2]:
                y_pro=1
            else:
                y_pro=2
            if y_pro==y_hat[i]:
                score=Score(y_hat[i],y_hat_pro[i])
            elif y_hat[i]==0:
                score=-0.3
            elif y_hat[i]==1:
                score=0.3
            else:
                score=0
            s.append(int(score*100))
        result={}
        for i in range(len(sample_names)):
            #print(np.array(sample_names.index)[i],self.dic[y_hat[i]]) 
            result[np.array(sample_names.index)[i]]=[self.dic[y_hat[i]],s[i]]
        #self.show_result(sample_names,y_hat)
        return result
    def batch_wav_test(self,path_to_audios):
        df_data,y_label,sample_names=self.batch_pitchs_and_labels(path_to_audios)
        df_data1,y_label1,sample_names1=self.batch_short_and_labels(path_to_audios)
        #print(sample_names,sample_names1)
        self.selector_f0.set_timeseries_container(df_data)
        test_features=self.selector_f0.transform(sample_names)
        
        
        self.selector_short.set_timeseries_container(df_data1)
        test_features1=self.selector_short.transform(sample_names1)
        
        
        test_features=pd.merge(test_features,test_features1,how= 'inner',left_index=True,right_index=True)
        test_features = self.scaler.transform(test_features)
        
        y_hat=self.clf.predict(test_features)
        y_hat_pro=np.array(self.clf.predict_proba(test_features))
        y_hat_pro=np.around(y_hat_pro, decimals=2)
        
        s=[]

        for i in range(len(y_hat_pro)):
            if y_hat_pro[i][0]>y_hat_pro[i][1] and y_hat_pro[i][0]/y_hat_pro[i][2]>1:
                y_pro=0
            elif y_hat_pro[i][1]>y_hat_pro[i][0] and y_hat_pro[i][1]>y_hat_pro[i][2]:
                y_pro=1
            else:
                y_pro=2
            if y_pro==y_hat[i]:
                score=Score(y_hat[i],y_hat_pro[i])
            elif y_hat[i]==0:
                score=-0.3
            elif y_hat[i]==1:
                score=0.3
            else:
                score=0
            s.append(int(score*100))        
        result=pd.DataFrame()
        result['sample']=sample_names.index
        result['pred']=pd.Series(y_hat).apply(lambda x : self.dic[x])
        result['score']=s
        csv_path=path_to_audios+'predictions.csv'
        result.to_csv(csv_path,index=False)
        print('result saved in '+csv_path)
        #self.show_result(sample_names,y_hat)
        result={}
        for i in range(len(sample_names)):
            #print(np.array(sample_names.index)[i],self.dic[y_hat[i]]) 
            result[np.array(sample_names.index)[i]]=[self.dic[y_hat[i]],s[i]]
        #self.show_result(sample_names,y_hat)
        return result
    def test(self,path):
        if os.path.isdir(path):
            return self.batch_wav_test(path)
        elif path.endswith('.wav'):
            return self.single_wav_test(path)
        else:
            #print('path error')
            raise ValueError (path+' is not a .wav file nor a path.')


# In[ ]:


#测试用
if __name__ =='__main__':
    c=predictor()
    path_one_wav='1603355245.wav'
    result=c.test(path_one_wav)
    print(result)
    
    path_wavs='/home/wangce/tmp_files/'
    result=c.test(path_wavs)
    print(result)
