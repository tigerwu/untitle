#!/bin/python
# -*- coding: utf-8 -*-

import sys
import importlib
import wave


try:
    importlib.reload(sys)
    sys.setdefaultencoding('utf-8')
except:
    pass

def check_wave(wavfile):
    f = None
    try:
        f = wave.open(wavfile, "rb")
        params = f.getparams()
        print(type(params))
        print(params)
        channels = f.getnchannels()
        print(channels)
        time_count = f.getparams().nframes / f.getparams().framerate
        print(time_count)
    except Exception as e:
        print('except:', e)
    finally:
        if f != None:
            f.close()
    pass

if __name__ == '__main__':
    check_wave("no.wav")
    check_wave("sound.wav")
    check_wave("bad.wav")
    check_wave("error.wav")