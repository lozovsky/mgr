#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import _pickle as pickle
import numpy
import tensorflow as tf
from tensorflow.core.protobuf import saver_pb2
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
#config.gpu_options.per_process_gpu_memory_fraction = 0.5
session = tf.Session(config=config)
import datetime
import dateutil.relativedelta
import time
import subprocess

import tflib
import tflib.linear
import tflib.conv1d

# Kolory bash
class bcolors:
    BLUE   = '\033[94m'
    GREEN  = '\033[92m'
    ORANGE = '\033[93m'
    RED    = '\033[91m'
    ENDC   = '\033[0m'

# Parsowanie argumentow
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',
                        type=str,
                        required=True,                    
                        help='Sciezka do folderu z wytrenowanymi danymi')
    parser.add_argument('--model',
                        type=str,
                        required=True,                    
                        help='Sciezka do pliku checkpoint')
    parser.add_argument('--output',
                        type=str,
                        required=True,
                        help='Sciezka do folderu wynikowego')
    parser.add_argument('--amount',
                        type=int,
                        default=1000000,
                        help='Liczba hasel do wygenerowania')
    parser.add_argument('--batch-size',
                        type=int,
                        default=64,
                        help='Batch size: liczba hasel ze zbioru treningowego rozprzestrzenianych do GAN w kazdej iteracji')
    parser.add_argument('--layer',
                        type=int,
                        default=5,
                        help='Liczba warstw szczatkowych dla generatora')
    parser.add_argument('--dimensionality',
                        type=int,
                        default=128,
                        help='Liczba wymiarow (wag) dla kazdej warstwy konwolucyjnej')
    parser.add_argument('--output-length',
                        type=int,
                        default=10,
                        help='Dlugosc generowanych hasel')
    parser.add_argument('--noise-size',
                        type=int,
                        default=128,
                        help='Rozmiar wektoru szumu - ile losowych bitow dodawanych jest do wejscia dla G')
    parser.add_argument('--gpu',
                        type=str,
                        default=0,
                        help='Indeks karty graficznej w systemie')
    args = parser.parse_args()
    return args

# Softmax
def softmax(logits, num_classes):
    return tf.reshape(
        tf.nn.softmax(
            tf.reshape(logits, [-1, num_classes])
        ),
        tf.shape(logits)
    )

# Blok szczatkowy
def ResBlock(name=None, inputs=None, dim=None):
    return inputs + (0.3*tflib.conv1d.Conv1D(name+'.2', dim, dim, 5, tf.nn.relu(tflib.conv1d.Conv1D(name+'.1', dim, dim, 5, tf.nn.relu(inputs)))))

# Generator
def Generator(noise_size=None, batch_size=None, output_length=None, dimensionality=None, layer=None, output_dim=None):
    if noise_size is None or batch_size is None or output_length is None or dimensionality is None or output_dim is None:
        raise ValueError('Co najmniej jeden z argumentow mial wartosc None!')
    output = tf.reshape(tflib.linear.Linear('Generator.Input', 128, output_length * dimensionality, tf.random_normal(shape=[batch_size, noise_size])), [-1, dimensionality, output_length])
    for i in range(1, layer):
        output = ResBlock('Generator.%d' % i, output, dimensionality)
    output = softmax(tf.transpose(tflib.conv1d.Conv1D('Generator.Output', dimensionality, output_dim, 1, output), [0, 2, 1]), output_dim)
    return output

# Generuj sample z zadanego modelu
def generate_samples(fake_inputs=None, char_map_inv=None, session=None):
    if fake_inputs is None or char_map_inv is None or session is None:
        raise ValueError('Co najmniej jeden z argumentow mial wartosc None!')
    
    samples = session.run(fake_inputs)
    samples = numpy.argmax(samples, axis=2)
    decoded_samples = []
    for i in range(0, len(samples)):
        decoded = []
        for j in range(0, len(samples[i])):
            decoded.append(char_map_inv[samples[i][j]])
        decoded_samples.append(tuple(decoded))
    return decoded_samples

def save(samples=None, opath=None, placeholder=None):
    if samples is None or opath is None or placeholder is None:
        raise ValueError('Co najmniej jeden z argumentow mial wartosc None!')
    
    with open(opath, 'a') as f:
        f.writelines([str(''.join(sample).replace(placeholder, '') + "\n") for sample in samples])
        
def runThread(args, num, gpu_id):
    if args is None:
        raise ValueError('Brak argumentow!')
    
    # Reset tensorflow, zacznij logowanie
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    with tf.device('/cpu:' + gpu_id):
        tf.reset_default_graph()
        
        # Placeholder - dopasowanie dlugosci hasla
        PLACEHOLDER="'"
            
        # Load charmaps
        with open(os.path.join(args.input, 'char_map.pickle'), 'rb') as f:
            char_map = pickle.load(f)
        with open(os.path.join(args.input, 'char_map_inv.pickle'), 'rb') as f:
            char_map_inv = pickle.load(f)

        fake_inputs = Generator(args.noise_size, args.batch_size, args.output_length, args.dimensionality, args.layer, len(char_map))
        
        # Wygeneruj sample
        with tf.Session() as session:
            saver = tf.train.Saver()
            saver.restore(session, args.model)
            sample_count = 0
            while sample_count < num:
                #for i in range(0, args.amount // args.batch_size):
                
                samples = generate_samples(fake_inputs, char_map_inv, session)
                samples = [entry for entry in list(set(samples)) if len("".join(entry).replace(PLACEHOLDER, '')) <= args.output_length and len("".join(entry).replace(PLACEHOLDER, '')) >= args.output_length / 2]
                sample_count = sample_count + len(samples)
                while sample_count > num:
                    samples = samples[:-1]
                    sample_count = sample_count - 1
                print("Generated %d%s" % (len(samples), ' ' * 20), end="\r")
                save(samples, args.output, PLACEHOLDER)

def run(args=None):
    if args is None:
        raise ValueError('Brak argumentow!')
    
    gpus = 0 
    from tensorflow.python.client import device_lib
    devices = [x.name.split(':')[-1] for x in device_lib.list_local_devices() if x.device_type == 'GPU']    
    if gpus == 0:
        runThread(args, args.amount, str(0))
    else:
        threads = []
        num_per_thread = args.amount // len(gpus)
        count = 1
        for gpu in gpus:
            num_samples = args.amount if (count == len(gpus)) else num_per_thread
            args.amount = args.amount - num_samples
            print("Starting thread on GPU #%s for %d samples" % (gpu, num_samples))
            p = subprocess.Popen([
                                    'env/bin/python3',
                                    'generate.py',
                                    '--input',
                                    str(args.input),
                                    '--model',
                                    str(args.model),
                                    '--output',
                                    str(args.output),
                                    '--amount',
                                    str(num_samples),
                                    '--batch-size',
                                    str(args.batch_size),
                                    '--layer',
                                    str(args.layer),
                                    '--dimensionality',
                                    str(args.dimensionality),
                                    '--output-length',
                                    str(args.output_length),
                                    '--noise-size',
                                    str(args.noise_size),
                                    '--gpu',
                                    str(gpu)
                                 ])
            threads.append(p)
            count = count + 1
        
        while threads:
            if threads[0].poll() is None:
                time.sleep(5)
            else: 
                del threads[0]
    


if __name__ == '__main__':
    start_time = datetime.datetime.fromtimestamp(time.time())
    run(parse_args())
    end_time   = datetime.datetime.fromtimestamp(time.time())
    rd = dateutil.relativedelta.relativedelta (end_time, start_time)
    print("Generowanie ukonczone po: %d ddniach, %d godzinach, %d minutach, %d sekundach" % (rd.days, rd.hours, rd.minutes, rd.seconds))
