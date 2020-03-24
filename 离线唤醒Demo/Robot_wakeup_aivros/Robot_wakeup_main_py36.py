# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 06:41:13 2019

@author: Aivros
"""

import pyaudio
import wave
import os
import collections
import contextlib
import sys
import webrtcvad
import difflib
import platform as plat
from SpeechModel251 import ModelSpeech
from LanguageModel2 import ModelLanguage
#import tensorflow as tf
import numpy as np
from pypinyin import lazy_pinyin
#------------------------------------------------------------------------------
#流程说明

#------------------------------------------------------------------------------





#1.多轮对话演示
#------------------------------------------------------------------------------
def wakeup(wakeup_path):
    chunk = 1024
    with wave.open(wakeup_path) as f:
        params = f.getparams()
        nchannels, sample_width, sample_rate, nframes = params[0:4]
       
        # 打开声卡设备
        pdev = pyaudio.PyAudio()
        stream = pdev.open(format = pdev.get_format_from_width(sample_width),
                           channels = nchannels,
                           rate = sample_rate,
                           output = True)
        
        # 读出数据并写到声卡
        while True:
            data = f.readframes(chunk)
            # 将数据转换为数组
            # r_data = np.fromstring(data, dtype=np.short)
            if len(data) == 0:
                break
            stream.write(data)
        
        # 关闭声卡设备
        stream.close()
        pdev.terminate()
    
    print('主人您好!请问有什麽吩咐？')
  
    input_restar = input("\nplease input 'y' to restar:\n").lower()
    if input_restar == 'y':
        print ('\nWelcome back!\n')
        
        return
    else:
        sys.exit(1)
        return
    
    return
#------------------------------------------------------------------------------
#2.唤醒检测
#------------------------------------------------------------------------------
def CTC_TTF():
    #ttf识别模型
    pass


def CTC_tf(current_path):
    
    datapath = ''
    modelpath = 'model_speech'
    
    system_type = plat.system() # 由于不同的系统的文件路径表示不一样，需要进行判断
    if(system_type == 'Windows'):
    	datapath = current_path
    	modelpath = modelpath + '\\'
    elif(system_type == 'Linux'):
    	datapath = 'dataset'
    	modelpath = modelpath + '/'
    else:
    	print('*[Message] Unknown System\n')
    	datapath = 'dataset'
    	modelpath = modelpath + '/'
    
    ms = ModelSpeech(datapath)
    
    ms.LoadModel(modelpath + 'speech_model251_e_0_step_12000.model')
    
    #ms.TestModel(datapath, str_dataset='test', data_count = 64, out_report = True)
    
    rr = ms.RecognizeSpeech_FromFile(current_path + '\\chunk-00.wav')
    
    print('*[提示] 语音识别结果：\n',rr)
    
    
    ml = ModelLanguage('model_language')
    ml.LoadModel()
    
    #str_pinyin = ['zhe4','zhen1','shi4','ji2', 'hao3','de5']
    #str_pinyin = ['jin1', 'tian1', 'shi4', 'xing1', 'qi1', 'san1']
    #str_pinyin = ['ni3', 'hao3','a1']
    str_pinyin = rr
    #str_pinyin =  ['su1', 'bei3', 'jun1', 'de5', 'yi4','xie1', 'ai4', 'guo2', 'jiang4', 'shi4', 'ma3', 'zhan4', 'shan1', 'ming2', 'yi1', 'dong4', 'ta1', 'ju4', 'su1', 'bi3', 'ai4', 'dan4', 'tian2','mei2', 'bai3', 'ye3', 'fei1', 'qi3', 'kan4', 'zhan4']
    r = ml.SpeechToText(str_pinyin)
    print('语音转文字结果：\n',r)
    
    ctc_result = hanzi_pinyin(r)
    
    return ctc_result


'''
def CTC_tf(vad_wav):
    #TF_CTC词识别模型

    from utils import decode_ctc, GetEditDistance
    
    
    # 0.准备解码所需字典，参数需和训练一致，也可以将字典保存到本地，直接进行读取
    from utils import get_data, data_hparams
    data_args = data_hparams()
    train_data = get_data(data_args)
    
    
    # 1.声学模型-----------------------------------
    from model_speech.cnn_ctc import Am, am_hparams
    
    am_args = am_hparams()
    am_args.vocab_size = len(train_data.am_vocab)
    am = Am(am_args)
    print('loading acoustic model...')
    am.ctc_model.load_weights('logs_am/model.h5')
    
    # 2.语言模型-------------------------------------------
    from model_language.transformer import Lm, lm_hparams
    
    lm_args = lm_hparams()
    lm_args.input_vocab_size = len(train_data.pny_vocab)
    lm_args.label_vocab_size = len(train_data.han_vocab)
    lm_args.dropout_rate = 0.
    print('loading language model...')
    lm = Lm(lm_args)
    sess = tf.Session(graph=lm.graph)
    with lm.graph.as_default():
        saver =tf.train.Saver()
    with sess.as_default():
        latest = tf.train.latest_checkpoint('logs_lm')
        saver.restore(sess, latest)
    
    # 3. 准备测试所需数据， 不必和训练数据一致，通过设置data_args.data_type测试，
    #    此处应设为'test'，我用了'train'因为演示模型较小，如果使用'test'看不出效果，
    #    且会出现未出现的词。
    data_args.data_type = 'train'
    data_args.shuffle = False
    data_args.batch_size = 1
    test_data = get_data(data_args)
    
    # 4. 进行测试-------------------------------------------
    am_batch = test_data.get_am_batch()
    word_num = 0
    word_error_num = 0
    for i in range(10):
        print('\n the ', i, 'th example.')
        # 载入训练好的模型，并进行识别
        inputs, _ = next(am_batch)
        x = inputs['the_inputs']
        y = test_data.pny_lst[i]
        result = am.model.predict(x, steps=1)
        # 将数字结果转化为文本结果
        _, text = decode_ctc(result, train_data.am_vocab)
        text = ' '.join(text)
        print('文本结果：', text)
        print('原文结果：', ' '.join(y))
        with sess.as_default():
            text = text.strip('\n').split(' ')
            x = np.array([train_data.pny_vocab.index(pny) for pny in text])
            x = x.reshape(1, -1)
            preds = sess.run(lm.preds, {lm.x: x})
            label = test_data.han_lst[i]
            got = ''.join(train_data.han_vocab[idx] for idx in preds[0])
            
            ctc_result = list(train_data.han_vocab[idx] for idx in preds[0])
            
            print('原文汉字：', label)
            print('识别结果：', got)
            word_error_num += min(len(label), GetEditDistance(label, got))
            word_num += len(label)
    print('词错误率：', word_error_num / word_num)
    sess.close()
    
    ctc_result = pypinyin(ctc_result)
    
    return ctc_result

'''

def check_ctc(list_a,list_b):
    #唤醒词拼音与识别词拼音比较，四个字母序列匹配，3个正确。
    o = 0
    #dict_list = [i for i in dict_b.values()]
    print(list_a,list_b)
    for i in list_a:
        print(i)
        if i in list_b:
            o +=1
        else:
            None
    print(o)
    if o >= 2:
        return True
    else:
        return False
    return 

#------------------------------------------------------------------------------            
#3.检测分割           
#------------------------------------------------------------------------------

class Frame(object):
    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration


class VAD(Frame,object):
    def __init__(self, parent=None):

        self.parent = parent

    def read_wave(self,path):
        with contextlib.closing(wave.open(path, 'rb')) as wf:
            num_channels = wf.getnchannels()
            assert num_channels == 1
            sample_width = wf.getsampwidth()
            assert sample_width == 2
            sample_rate = wf.getframerate()
            assert sample_rate in (8000, 16000, 32000)
            pcm_data = wf.readframes(wf.getnframes())
            return pcm_data, sample_rate
    
    
    def write_wave(self,path, audio, sample_rate):
        with contextlib.closing(wave.open(path, 'wb')) as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(audio)
    
    

    
    def frame_generator(self,frame_duration_ms, audio, sample_rate):
        n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
        offset = 0
        timestamp = 0.0
        duration = (float(n) / sample_rate) / 2.0
        while offset + n < len(audio):
            yield Frame(audio[offset:offset + n], timestamp, duration)
            timestamp += duration
            offset += n
    
    
    def vad_collector(self,sample_rate, frame_duration_ms,
                      padding_duration_ms, vad, frames):
        num_padding_frames = int(padding_duration_ms / frame_duration_ms)
        ring_buffer = collections.deque(maxlen=num_padding_frames)
        triggered = False
        voiced_frames = []
        for frame in frames:
            sys.stdout.write(
                '1' if vad.is_speech(frame.bytes, sample_rate) else '0')
            if not triggered:
                ring_buffer.append(frame)
                num_voiced = len([f for f in ring_buffer
                                  if vad.is_speech(f.bytes, sample_rate)])
                if num_voiced > 0.9 * ring_buffer.maxlen:
                    sys.stdout.write('+(%s)' % (ring_buffer[0].timestamp,))
                    triggered = True
                    voiced_frames.extend(ring_buffer)
                    ring_buffer.clear()
            else:
                voiced_frames.append(frame)
                ring_buffer.append(frame)
                num_unvoiced = len([f for f in ring_buffer
                                    if not vad.is_speech(f.bytes, sample_rate)])
                if num_unvoiced > 0.9 * ring_buffer.maxlen:
                    sys.stdout.write('-(%s)' % (frame.timestamp + frame.duration))
                    triggered = False
                    yield b''.join([f.bytes for f in voiced_frames])
                    ring_buffer.clear()
                    voiced_frames = []
        if triggered:
            sys.stdout.write('-(%s)' % (frame.timestamp + frame.duration))
        sys.stdout.write('\n')
        if voiced_frames:
            yield b''.join([f.bytes for f in voiced_frames])
    
    
    def main(self,wav_path,current_path):
        #if len(args) != 2:
            #sys.stderr.write(
                #'Usage: example.py <aggressiveness> <path to wav file>\n')
            #sys.exit(1)
        audio, sample_rate = self.read_wave(wav_path)
        vad = webrtcvad.Vad(int(1))
        frames = self.frame_generator(30, audio, sample_rate)
        frames = list(frames)
        segments = self.vad_collector(sample_rate, 30, 300, vad, frames)
        print(current_path)
        #ctc_read_path = current_path + '\\test\\vad_1.wav'
        select_sum = 0
        for i, segment in enumerate(segments):
            vad_wav = 'chunk-%002d.wav' % (i,)
            print(' Writing %s' % (vad_wav,))
            self.write_wave(vad_wav, segment, sample_rate)
    
            vad_wav_p = 'chunk-%002d.wav' % (i,)
            print('--end')
            select_sum +=1
            self.write_wave(vad_wav_p, segment, sample_rate)
        #print(audio, sample_rate,args[1],segments)
        return select_sum


           
#------------------------------------------------------------------------------       
#4.循环录制

#------------------------------------------------------------------------------
def recording():
    pass


def loop_recording(current_path,text_pinyin_list):
    sum = 0
    while True:
        input_filename = "wakeup.wav"               # 麦克风采集的语音输入
        input_filepath = current_path             # 输入文件的path
        wav_path = input_filepath + '\\' + input_filename
        wakeup_path =  current_path + '\\' + 'online_star.wav'
        print(wav_path)
    
        CHUNK = 256
        FORMAT = pyaudio.paInt16
        CHANNELS = 1                # 声道数
        RATE = 16000                # 采样率
        RECORD_SECONDS = 2
        WAVE_OUTPUT_FILENAME = wav_path #目标文件目录
        p = pyaudio.PyAudio()

        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)


        
        frames = []
        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(data)
            print("-"*i)
        print("*"*10, "录音结束\n")

        stream.stop_stream()
        stream.close()
        p.terminate()

        wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()

        #waveform(in_path)
        
        vad = VAD()
        
        vad_check = vad.main(wav_path,current_path)
    
        
        if vad_check:
            ctc_tf_pinyin_list =  CTC_tf(current_path) #小优小优
            print(ctc_tf_pinyin_list)
            #ctc_tf_pinyin_dict = hanzi_pinyin(ctc_tf_word)
            
            check_ctc_tf = check_ctc(ctc_tf_pinyin_list,text_pinyin_list)
            print(sum)
            if check_ctc_tf or sum == 3:
                wakeup(wakeup_path)
            else:
                sum += 1
                
        else:
            None                            
    return 
  
#------------------------------------------------------------------------------
# 5.唤醒设置
#------------------------------------------------------------------------------
'''entrance: interactive_entry(). 
      
'''       
    
def wakeup_word():
   
    while True:

        message_h = "\n请输入唤醒词：\n"
        input_h = input(message_h) 

        if input_h >= '\u4e00' and input_h <= '\u9fa5' and len(input_h) == 4:

            word_list = list(input_h)
            
            return word_list
        else:
            print('\n您输入的唤醒词不符合规则， 请重新输入!\n')    
    return

 
    
def hanzi_pinyin(word_list):                           

    # 不考虑多音字的情况
    print(word_list)
    #pinyin_dict = {i: lazy_pinyin(val) for i, val in enumerate(word_list)}
    pinyin_list = lazy_pinyin(word_list) 
    print(pinyin_list)

    return pinyin_list


def interactive_entry():
    '''
    'User-interface' entrance.
    '''
    input_restar = input("\n唤醒词设置规则请输入 'y':\n").lower()
    if input_restar == 'y':
        print ('\n唤醒词设置规则!\n')
        #唤醒词规则
        message = "唤醒词设置分为两步：\n\
                   1.四个字的唤醒词设置\n\
                   2.与文字相同的语音录入！\n"
                         
        print(message)
        #唤醒词录入
        wakeup_list = wakeup_word() 
        #汉字转拼音
        text_pinyin_list = hanzi_pinyin(wakeup_list)
        print(text_pinyin_list)
        
        return text_pinyin_list
        
        #用户声音录入
        #usr_audio = recording()
        #用户唤醒词语料库生成
        #wakeup_audio_data = imgaug()

    else:
        
        wakeup_list = ['小', '优','小','优']
        text_pinyin_dict = hanzi_pinyin(wakeup_list)        
        return text_pinyin_dict
        
    return 
#------------------------------------------------------------------------------    
#6.Anytime exit.
#------------------------------------------------------------------------------
def quit_sometime():
    quit = sys.stdin.read(1)
    if quit == '\x1b':
        sys.exit()
        return print ('Welcome again!')

#------------------------------------------------------------------------------
# 7.Main function.
#------------------------------------------------------------------------------
def main(): 
 
    message = "Hi ^_^, Welcome !\nPleas speaker something!\n"            
    print(message)
    while True:
        quit_sometime()
        text_pinyin_dict = interactive_entry()
        #try:
        current_path = os.getcwd()        
        loop_recording(current_path,text_pinyin_dict) 
        #except:
            #print('opps! I broke down. :( ')
            #return
        input_restar = input("\nplease input 'y' to restar:\n").lower()
        if input_restar == 'y':
            print ('\nWelcome back!\n')
            return
        else:
            sys.exit(1)
            return
        
    return print ('Welcome again!')
if __name__ == "__main__":
    main()
#------------------------------------------------------------------------------