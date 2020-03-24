# Robot_wakeup_aivros #



## Introduction 简介


本项目是基于离线环境的语音唤醒demo。

### Speech Model 语音模型

CNN + LSTM/GRU + CTC


### Language Model 语言模型

基于概率图的最大熵隐马尔可夫模型

输入为汉语拼音序列，输出为对应的汉字文本

## About Accuracy 关于准确率

当前，最好的模型在测试集上基本能达到80%的汉语拼音正确率

不过由于目前国际和国内的部分团队能做到98%，所以正确率仍有待于进一步提高

## Python Import
Python的依赖库

* python_speech_features
* TensorFlow
* Keras
* Numpy
* wave
* matplotlib
* math
* Scipy
* h5py
* http
* urllib   
* pyaudio
* os
* collections
* contextlib
* sys
* webrtcvad
* difflib
* platform
* pypinyin

