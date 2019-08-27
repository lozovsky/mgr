import argparse
import random
import sys
import os
import subprocess
import numpy
import collections
import _pickle as pickle
import tensorflow as tf
import datetime
import dateutil.relativedelta
import time
import logging
from threading import Thread
import multiprocessing
import math
import shutil

import tflib
import tflib.linear
import tflib.conv1d

# Domyslny poziom logowania - brak logowania
NOISE_LEVEL = None
# Zainicjuj podstawowa konfiguracje tensorflow
config = tf.ConfigProto()
# Dynamiczne przydzielanie zasobow karty graficznej
config.gpu_options.allow_growth = True
# Sztywny przydzial zasobow karty graficznej
# config.gpu_options.per_process_gpu_memory_fraction = 0.5
# Utworz sesje tensorflow
session = tf.Session(config=config)

# Kolory bash
class bcolors:
	BLUE = '\033[94m'
	GREEN = '\033[92m'
	ORANGE = '\033[93m'
	RED = '\033[91m'
	ENDC = '\033[0m'

# Parsowanie argumentow
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--training-data', type = str, required = True, help = 'Sciezka do danych treningowych (kodowanie UTF-8)')
    parser.add_argument(
                '--dictionary',
                type = str,
                help = 'Sciezka do slownika (kodowanie UTF-8)')
    parser.add_argument(
                '--output',
                type = str,
                required = True,
                help = 'Nazwa folderu z wynikami')
    parser.add_argument(
                '--save-iterations',
                type = int,
                default = 1000,
                help = 'Zapisz wyniki po n iteracjach (domyslnie 1000)')
    parser.add_argument(
                '--batch-size',
                type = int,
                default = 64,
                help = 'Batch size: liczba hasel ze zbioru treningowego rozprzestrzenianych do GAN w kazdej iteracji')
    parser.add_argument(
                '--iterations',
                type = int,
                default = 199000,
                help = 'Liczba iteracji (w kazdej iteracji wywolywany jest generator oraz zadana liczba wywolan dyskryminatora)')
    parser.add_argument(
                '--discriminator-iterations',
                type = int,
                default = 10,
                help = 'Liczba wywolan dyskryminatora w jednej iteracji')
    parser.add_argument(
                '--layer',
                type = int,
                default = 5,
                help = 'Liczba warstw szczatkowych dla generatora i dyskryminatora')
    parser.add_argument(
                '--dimensionality',
                type = int,
                default = 128,
                help = 'Liczba wymiarow (wag) dla kazdej warstwy konwolucyjnej')
    parser.add_argument(
                '--gradient-penalty',
                type = int,
                default = 10,
                help = 'Wspolczynnik poprawkowy aplikowany do gradientu dyskryminatora')
    parser.add_argument(
                '--output-length',
                type = int,
                default = 10,
                help = 'Maksymalna dlugosc generowanych przykladow')
    parser.add_argument(
                '--noise-size',
                type = int,
                default = 128,
                help = 'Rozmiar wektoru szumu - ile losowych bitow dodawanych jest do wejscia dla G')
    parser.add_argument(
                '--learning-rate',
                type = float,
                default = 0.001,
                help = 'Tempo dopasowywania wag w modelu')
    parser.add_argument(
                '--beta1',
                type = float,
                default = 0.5,
                help = 'Wspolczynnik beta1 - spadajace tempo obecnej sredniej gradientu')
    parser.add_argument(
                '--beta2',
                type = float,
                default = 0.9,
                help = 'Wspolczynnik beta2 - spadajace tempo obecnego kwadratu gradientu')
    parser.add_argument(
                '--gpu',
                type = int,
                default = 0,
                help = 'Indeks karty graficznej w systemie')
    parser.add_argument(
                '-v',
                action = 'store_true',
                help = 'Logowanie komunikatow typu CRITICAL')
    parser.add_argument(
                '-vv',
                action = 'store_true',
                help = 'Logowanie komunikatow typu CRITICAL, ERROR')
    parser.add_argument(
                '-vvv',
                action = 'store_true',
                help = 'Logowanie komunikatow typu CRITICAL, ERROR, WARNING')
    parser.add_argument(
                '-vvvv',
                action = 'store_true',
                help = 'Logowanie komunikatow typu CRITICAL, ERROR, WARNING, INFO')
    parser.add_argument(
                '-vvvvv',
                action = 'store_true',
                help = 'Logowanie komunikatow typu CRITICAL, ERROR, WARNING, INFO, DEBUG')
    args = parser.parse_args()
    return args

# Procedura trenowania
def run(args=None):
    global NOISE_LEVEL
    if args is None:
        raise ValueError('Nie podano wymaganych argumentow')

    # Utworz foldery
    setup_folders(args.output)

    # Ustal poziom logowania
    logging_level = set_log_level({
        0: args.v,
        1: args.vv,
        2: args.vvv,
        3: args.vvvv,
        4: args.vvvvv})
    # Utworz plik log w folderze logs sluzacy do ich przechowywania
    # Okresl format generowanych logow (timestamp, poziom, wiadomosc)
    logging.basicConfig(filename = os.path.join(args.output, 'logs', 'log'), filemode = 'a', format = '(%(asctime)s) [%(levelname)s]: %(message)s', level = logging_level)

    # Zaloguj zadane argumenty
    log("DEBUG", "Arguments: %s)" % (str(args)))

    # Okresl ID karty graficznej, jaka am zostac uzyta
    log("DEBUG", "ID karty graficznej: #%s" % (str(args.gpu)))
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # Placeholder, uzywany do sztucznego wydluzania hasel
    PLACEHOLDER = "'"
    log("DEBUG", "Placeholder == %s" % (PLACEHOLDER))

    # Analizuj dane treningowe, wygeneruj char maps
    log("DEBUG", "Analiza danych treningowych, generowanie char maps")
    passwords, char_map, char_map_inv = parse_file(pfile = args.training_data, dfile = args.dictionary, opath = args.output, olength = args.output_length, placeholder = PLACEHOLDER)
    # Zapisz zanalizowane dane
    save(opath = args.output, filtered_lines = passwords, char_map = char_map, char_map_inv = char_map_inv)

    # Reset tensorflow, zacznij logowanie
    log("DEBUG", "Poziom logowania TF: %s" % (os.environ['TF_CPP_MIN_LOG_LEVEL']))
    log("DEBUG", "Reset TensorFlow graph")
    tf.reset_default_graph()

    # Dyskretne dane rzeczywiste
    real_inputs_discrete = tf.placeholder(tf.int32, shape = [args.batch_size, args.output_length], name = "real_inputs_discrete")
    # Dane rzeczywiste
    real_inputs = tf.one_hot(real_inputs_discrete, len(char_map))
    # Dane zmyslone
    fake_inputs = Generator(args.noise_size, args.batch_size, args.output_length, args.dimensionality, args.layer, len(char_map))
    # Dyskryminator danych rzeczywistych
    disc_real = Discriminator(real_inputs, args.output_length, args.dimensionality, args.layer, len(char_map))
    # Dyskryminator danych zmyslonych
    disc_fake = Discriminator(fake_inputs, args.output_length, args.dimensionality, args.layer, len(char_map))
    # Koszt dyskryminatora
    disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)
    # Koszt generatora
    gen_cost = -tf.reduce_mean(disc_fake)
    # Wspolczynnik alpha
    alpha = tf.random_uniform(shape = [args.batch_size, 1, 1], minval = 0., maxval = 1.)
    # Roznice miedzy danymi zmyslonymi a rzeczywistymi
    differences = fake_inputs - real_inputs
    # Interpolacje
    interpolates = real_inputs + (alpha * differences)
    # Gradienty
    gradients = tf.gradients(Discriminator(interpolates, args.output_length, args.dimensionality, args.layer, len(char_map)), [interpolates])[0]
    # Slopes
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices = [1,2]))
    # Gradient penalty
    gradient_penalty = tf.reduce_mean((slopes - 1.)**2)
    # Zwiekszenie kosztu dyskryminatora
    disc_cost += args.gradient_penalty * gradient_penalty
    # Parametry generatora
    gen_params = tflib.params_with_name('Generator')
    # Parametry dyskryminatora
    disc_params = tflib.params_with_name('Discriminator')
    # Siec neuronowa generatora oraz
    # Wspolczynniki beta1, beta2 dla generatora
    gen_train_op = tf.train.AdamOptimizer(learning_rate = args.learning_rate, beta1 = args.beta1, beta2 = args.beta2).minimize(gen_cost, var_list = gen_params)
    # Siec neuronowa dyskryminatora oraz
    # Wspolczynniki beta1, beta2 dla dyskryminatora
    disc_train_op = tf.train.AdamOptimizer(learning_rate = args.learning_rate, beta1 = args.beta1, beta2 = args.beta2).minimize(disc_cost, var_list = disc_params)

    # Uruchom sesje tensorflow
    with tf.Session() as session:
        # Przywrocenie istniejacej sesji
        old_iteration = 0
        try:
            loader = tf.train.Saver()
            # Pobierz informacje o ostatnim istniejacym checkpoincie - folder saved
            files = [os.path.join(args.output, 'saved', f) for f in os.listdir(os.path.join(args.output, 'saved')) if 'model' in f]
            files.sort(key=lambda x: os.path.getmtime(x))
            if files:
                last_checkpoint = '.'.join(os.path.basename(files[-1]).split('.')[:-1])
                old_iteration = int(last_checkpoint.split('.')[0].split('_')[1])
                loader.restore(session, os.path.join(args.output, 'saved', last_checkpoint))
                log("INFO", "Zaladowano checkpoint: " + str(last_checkpoint))
        # W przypadku, gdy nie odnaleziono istniejacej sesji
        except Exception as e:
            log("WARNING", str(e))
            pass

        # Uruchom sesje
        session.run(tf.global_variables_initializer())
        gen = inf_train_gen(passwords, args.batch_size, char_map)

        # Zapisywanie logow tensorflow
        writer = tf.summary.FileWriter(os.path.join(args.output, 'logs'), graph = tf.get_default_graph())

        # Wykonanie zadanej/pozostalej liczby iteracji
        for iteration in range(old_iteration + 0, old_iteration + args.iterations):
            log("DEBUG", "Obecna iteracja: %d%s" % (iteration, ' ' * 20), "\r")

            # Trenowanie generatora
            if iteration > 0:
                _ = session.run(gen_train_op)

            # Trenowanie dyskryminatora
            for i in range(0, args.discriminator_iterations):
                _data = gen.__next__()
                _disc_cost, _ = session.run([disc_cost, disc_train_op], feed_dict = {real_inputs_discrete:_data})

            # Zapisz obecna sesje
            if iteration % args.save_intervall == 0 and iteration > 0:
                save(opath = args.output, session = session, sessionName = str(iteration))

                # Wygeneruj 50 przykladow
                samples = generate_samples(session, fake_inputs, char_map, char_map_inv)
                samples = [entry for entry in list(set(samples)) if len("".join(entry).replace(PLACEHOLDER, '')) <= args.output_length and len("".join(entry).replace(PLACEHOLDER, '')) >= args.output_length / 2]
                while len(samples) < 50:
                    samples = samples + generate_samples(session, fake_inputs, char_map, char_map_inv)
                    samples = [entry for entry in list(set(samples)) if len("".join(entry).replace(PLACEHOLDER, '')) <= args.output_length and len("".join(entry).replace(PLACEHOLDER, '')) >= args.output_length / 2]

                samples = samples[:49]
                # Zapisz wygenerowane przyklady
                with open(os.path.join(args.output, 'samples', 'samples_' + str(iteration) + '.txt'), 'w') as sample_file:
                    sample_file.writelines(["".join(sample).replace(PLACEHOLDER, '') + "\n" for sample in samples])

                log("INFO", "Zapisano iteracje: %u%s" % (iteration, ' ' * 20))
                log("INFO", "Wygenerowane przyklady: %s" % (', '.join(["".join(sample).replace(PLACEHOLDER, '') for sample in samples])))


# Utworz folder o zadanej nazwie, jesli jeszcze nie istnieje
def setup_folders(opath=None):
    # Sprawdz, czy podano wymagany argument
    if opath is None:
        raise ValueError('Nie podano nazwy folderu wynikowego')

    # Utworz folder glowny, jesli jeszcze nie istnieje
    if not os.path.isdir(opath):
        os.makedirs(opath)

    # Utworz subfolder ze slownikami, jesli jeszcze nie istnieje
    if not os.path.isdir(os.path.join(opath, 'dictionaries')):
        os.makedirs(os.path.join(opath, 'dictionaries'))

    # Utworz subfolder zawierajacy dane treningowe, jesli jeszcze nie istnieje
    if not os.path.isdir(os.path.join(opath, 'passwords')):
        os.makedirs(os.path.join(opath, 'passwords'))

    # Utworz subfolder z checkpointami, jesli jeszcze nie istnieje
    if not os.path.isdir(os.path.join(opath, 'saved')):
        os.makedirs(os.path.join(opath, 'saved'))

    # Utworz subfolder z wygenerowanymi samplami, jesli jeszcze nie istnieje
    if not os.path.isdir(os.path.join(opath, 'samples')):
        os.makedirs(os.path.join(opath, 'samples'))

    # Utworz subfolder z logami, jesli jeszcze nie istnieje
    if not os.path.isdir(os.path.join(opath, 'logs')):
        os.makedirs(os.path.join(opath, 'logs'))

# Obsluga poziomu logowania
def set_log_level(loglvl = None):
    global NOISE_LEVEL
    # args.vvvvv
    if loglvl[4]:
        logging_level = logging.DEBUG
        NOISE_LEVEL = 'DEBUG'
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
    # args.vvvv
    elif loglvl[3]:
        logging_level = logging.INFO
        NOISE_LEVEL = 'INFO'
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    # args.vvv
    elif loglvl[2]:
        logging_level = logging.ERROR
        NOISE_LEVEL = 'WARNING'
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    # args.vv
    elif loglvl[1]:
        logging_level = logging.ERROR
        NOISE_LEVEL = 'ERROR'
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    # args.v
    elif loglvl[0]:
        logging_level = logging.CRITICAL
        NOISE_LEVEL = 'CRITICAL'
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'
    # Nie loguj nic
    else:
        logging_level = None
        NOISE_LEVEL = None
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'
    return logging_level

# Procedura logowania wiadomosci
def log(LEVEL = None, MSG = None, LINEEND = None):
    global NOISE_LEVEL

    # Poziomy logowania
    levels = {
            'debug': 0,
            'info': 1,
            'warning': 2,
            'error': 3,
            'critical': 4}

    # Nie loguj, jesli nie ma podanego poziomu, wiadomosci, badz logowanie jest dezaktywowane
    if LEVEL is None or MSG is None or NOISE_LEVEL is None:
        return

    # Loguj w zaleznosci o zadanego poziomu
    if LEVEL.lower() == 'debug':
        logging.debug(MSG.strip())
    elif LEVEL.lower() == 'info':
        logging.info(MSG.strip())
    elif LEVEL.lower() == 'warning':
        logging.warning(MSG.strip())
    elif LEVEL.lower() == 'error':
        logging.error(MSG.strip())
    elif LEVEL.lower() == 'critical':
        logging.critical(MSG.strip())

    # Wyswietl w ekranie konsoli logowane rzeczy
    if levels[NOISE_LEVEL.lower()] <= levels[LEVEL.lower()]:
        print('(%s) [%s] %s' % (time.asctime(), LEVEL, MSG), end = LINEEND if LINEEND else '\n')

# Zaladuj dane treningowe - odrzuc zbyt dlugie hasla
def parse_file(pfile = None, dfile = None, opath = None, olength = None, placeholder = None):
    # Sprawdz, czy zostaly podane argumenty funkcji
    if pfile is None or olength is None or placeholder is None:
        raise ValueError('Nie udalo sie uzyskac jednego lub wiecej: pfile, olength, placeholder')

    # Sprawdz, czy sciezka do danych treningowych jest prawidlowa
    if not os.path.isfile(pfile):
        log("CRITICAL", "Nie odnaleziono pliku z danymi treningowymi")
        sys.exit(1)

    changed = [True, False]

    # Zaladuj hasla z subfolderu passwords, jesli istnieja
    if os.path.isfile(os.path.join(opath, 'passwords', os.path.basename(pfile))):
        passwords = load(opath = opath, passwords = True)
        changed[0] = False
    # Skopiuj hasla do subfolderu passwords
    else:
        shutil.copy2(pfile, os.path.join(opath, 'passwords', os.path.basename(pfile)))
        passwords = []
        try:
            # Pobierz liczbe linii (liczbe hasel)
            num_lines = int(subprocess.check_output(['/usr/bin/wc', '-l', pfile]).split()[0])
        except:
            # Nie znaleziono systemowego programu wc
            log("CRITICAL", "Nie odnaleziono programu wc - /usr/bin/wc")
            sys.exit(1)
        count_accepted = 0
        count_skipped = 0

        # Pobierz hasla z pliku
        with open(pfile, 'r', encoding='utf-8') as training_file:
            for password in training_file:
                password = password.strip()
                if password: # Pomin puste linie
                    if not password.isdigit(): # Pomin linie z samymi cyframi
                        password = tuple(password)
                        if len(password) <= olength: # Zaakceptuj haslo o dobrej dlugosci
                            passwords.append(password + ( (placeholder,)*(olength - len(password))))
                            count_accepted = count_accepted + 1
                        else: # Odrzuc zbyt dlugie haslo
                            count_skipped = count_skipped + 1
                        # Zaloguj postep
                        log("DEBUG", "Analiza hasel: %s%d (Zaakceptowane)%s / %s%d (Odrzucone) %s / %s%d (Total)%s %s" % (bcolors.GREEN, count_accepted, bcolors.ENDC, bcolors.ORANGE, count_skipped, bcolors.ENDC, bcolors.BLUE, num_lines, bcolors.ENDC, ' ' * 20), "\r")
                    else:
                        count_skipped = count_skipped + 1
                else:
                    count_skipped = count_skipped + 1
        log("DEBUG", "Analiza hasel: %s%d (Zaakceptowane)%s / %s%d (Odrzucone) %s / %s%d (Total)%s %s" % (bcolors.GREEN, count_accepted, bcolors.ENDC, bcolors.ORANGE, count_skipped, bcolors.ENDC, bcolors.BLUE, num_lines, bcolors.ENDC, ' ' * 20), "\r")
        save(opath = opath, passwords = passwords)

    # Obsluga slownikow
    dictionaries = {}
    if dfile:
        files = [dfile] if os.path.isfile(dfile) else [f for f in os.listdir(dfile) if os.path.isfile(os.path.join(dfile, f))]
        if [f for f in files if not os.path.isfile(os.path.join(opath, 'dictionaries', os.path.basename(f)))]:
            log("DEBUG", "Odnaleziony slownik: %s" % (', '.join(files)))
            changed[1] = True
            tlist = []
            for file in files:
                shutil.copy2(file, os.path.join(opath, 'dictionaries', os.path.basename(file)))
                t = Thread(target = parse_dictionary, args=(dfile, file, olength, dictionaries))
                tlist.append(t)
                t.start()
            while tlist:
                tlist[0].join()
                tlist.remove(tlist[0])
            save(opath = opath, dictionaries = dictionaries)
        else:
            dictionaries = load(opath = opath, dictionaries = True)

    # Wymieszaj kolejnosc hasel
    numpy.random.shuffle(passwords)

    # Gdy nie ma zmian, zaladuj istniejace juz dane
    if not changed[0] and not changed[1]:
        filtered_lines = load(opath = opath, filtered_lines = True)
        numpy.random.shuffle(filtered_lines)
        char_map = load(opath = opath, char_map = True)
        char_map_inv = load(opath = opath, char_map_inv = True)
    else: # Sprawdz, czy w slowniku znajduja sie slowa
        finished_passwords = []
        if dfile:
            found = []
            count = 0
            word_found = 0

            def check_dictionary(password, placeholder, dictionaries, found, finished_passwords):
                temp_pw = password
                for dictionary in dictionaries:
                    for word in dictionaries[dictionary]:
                        if word in ''.join(tmp_pw).replace(placeholder, '') and len(word) > 1:
                            tmp_pw = tuple(''.join(tmp_pw).replace(word, ''))
                            found.append(word)
                finished_passwords.append(tmp_pw + ( (placeholder,)*(olength - len(tmp_pw))))

            tlist = []
            batches = 1
            for password in passwords:
                t = Thread(target = check_dictionary, args=(password, placeholder, dictionaries, found, finished_passwords))
                tlist.append(t)
                if len(tlist) >= multiprocessing.cpu_count() * 2:
                    log("DEBUG", "Sprawdzanie slownikow przez %d watkow. Batch: #%d/%d" % (len(tlist), batches, mat.ceil(len(passwords)/multiprocessing.cpu_count()/2)), "\r")
                    for entry in tlist:
                        entry.start()
                    while tlist:
                        tlist[0].join()
                        tlist.remove(tlist[0])
                    batches = batches + 1
            log("DEBUG", "Sprawdzanie slownikow przez %d watkow. Batch: #%d/%d" % (len(tlist), batches, mat.ceil(len(passwords)/multiprocessing.cpu_count()/2)), "\r")
            for entry in tlist:
                entry.start()
            while tlist:
                tlist[0].join()
                tlist.remove(tlist[0])
        else:
            fount = []

        # Obsluga charow
        words = [f for f in finished_passwords if ''.join(f).replace(placeholder, '') != ''] + [tuple(f) + ( (placeholder,)*(olength - len(f))) for f in found] if finished_passwords else passwords
        counts = collections.Counter(char for password in words for char in password)
        char_map = {'unk': 0}
        char_map_inv = ['unk']
        counts = counts + collections.Counter(word for word in found) # Zlaczenie list slow i charow
        for char, count in counts.most_common(2048): # Pozwol na uczenie sie z wielu zrodel
            if char not in char_map:
                char_map[char] = len(char_map_inv)
                char_map_inv.append(char)

        filtered_lines = []
        for password in words:
            filtered_line = []
            for char in password:
                if char in char_map:
                    filtered_line.append(char)
                else:
                    filtered_line.append('unk')
            filtered_lines.append(tuple(filtered_line))
    return filtered_lines, char_map, char_map_inv

# Zmapuj dane ze slownika, jezeli zostal podany
def parse_dictionary(path = None, file = None, olength = 0, dictionary = None):
    entries = []
    file_path = file if os.path.isfile(file) else os.path.join(path, file)
    with open(file_path, 'r') as dictionary:
        entries = dictionary.read().splitlines()
        # Filtrowanie slow dluzszych niz zadana dlugosc
        entries = [entry for entry in entries if len(entry) <= olength]
        entries.sort(key = lambda s: len(s))
        entries = list(reversed(entries))
        dictionary[''.join(file.split('.')[:-1])] = entries

# Softmax
def softmax(logits, num_classes):
    return tf.reshape(
            tf.nn.softmax(
                tf.reshape(logits, [-1, num_classes])
            ),
            tf.shape(logits)
    )

# Blok szczatkowy
def ResBlock(name = None, inputs = None, dim = None):
    return inputs + (0.3 * tflib.conv1d.Conv1D(name + '.2', dim, dim, 5, tf.nn.relu(tflib.conv1d.Conv1D(name + '.1', dim, dim, 5, tf.nn.relu(inputs)))))

# Generator
def Generator(noise_size = None, batch_size = None, output_length = None, dimensionality = None, layer = None, output_dim = None):
    if noise_size is None or batch_size is None or output_length is None or dimensionality is None or output_length is None:
        raise ValueError('Generator - niewystarczajaca liczba argumentow')
    output = tf.reshape(tflib.linear.Linear('Generator.Input', 128, output_length * dimensionality, tf.random_normal(shape = [batch_size, noise_size])), [-1, dimensionality, output_length])
    for i in range(1, layer):
        output = ResBlock('Generator.%d' % i, output, dimensionality)
    output = softmax(tf.transpose(tflib.conv1d.Conv1D('Generator.Output', dimensionality, output_dim, 1, output), [0, 2, 1]), output_dim)
    return output

# Dyskryminator
def Discriminator(inputs, output_length = None, dimensionality = None, layer = None, input_dim = None):
    output = tflib.conv1d.Conv1D('Discriminator.Input', input_dim, dimensionality, 1, tf.transpose(inputs, [0,2,1]))
    for i in range(1, layer):
        output = ResBlock('Discriminator.%d' % i, output, dimensionality)
    output = tflib.linear.Linear('Discriminator.Output', output_length * dimensionality, 1, tf.reshape(output, [-1, output_length * dimensionality]))
    return output

# Ladowanie danych z ostatniej sesji
def load(opath = None, passwords = None, dictionary = None, filtered_lines = None, char_map = None, char_map_inv = None, session = None):
    if opath is None:
        raise ValueError('load - brak opath')

    if passwords:
        if os.path.isfile(os.path.join(opath, 'passwords.pickle')):
            with open(os.path.join(opath, 'passwords.pickle'), 'rb') as f:
                passwords = pickle.load()
                log("DEBUG", "Zaladowano hasla")
        return passwords

    if dictionary:
        if os.path.isfile(os.path.join(opath, 'dictionaries.pickle')):
            with open(os.path.join(opath, 'dictionaries.pickle'), 'rb') as f:
                dictionaries = pickle.load()
                log("DEBUG", "Zaladowano slowniki")

        return dictionaries

    if filtered_lines:
        if os.path.isfile(os.path.join(opath, 'filtered_lines.pickle')):
            with open(os.path.join(opath, 'filtered_lines.pickle'), 'rb') as f:
                filtered_lines = pickle.load(f)
                log("DEBUG", "Zaladowano przefiltrowane dane")
        return filtered_lines

    if char_map:
        if os.path.isfile(os.path.join(opath, 'char_map.pickle')):
            with open(os.path.join(opath, 'char_map.pickle'), 'rb') as f:
                char_map = pickle.load()
                log("DEBUG", "Zaladowano mape charow")
        return char_map

    if char_map_inv:
        if os.path.isfile(os.path.join(opath, 'char_map_inv.pickle')):
            with open(os.path.join(opath, 'char_map_inv.pickle'), 'rb') as f:
                char_map_inv = pickle.load(f)
                log("DEBUG", "Zaladowano odwrocona mape charow")
        return char_map_inv

# Zapisywanie sesji
def save(opath = None, passwords = None, dictionaries = None, filtered_lines = None, char_map = None, char_map_inv = None, session = None):
    if opath is None:
        raise ValueError('save - brak opath')

    if passwords:
        with open(os.path.join(opath, 'passwords.pickle'), 'wb') as f:
            pickle.dump(passwords, f)
            log("DEBUG", "Zapisano hasla")

    if dictionaries:
        with open(os.path.join(opath, 'dictionaries.pickle'), 'wb') as f:
            pickle.dump(dictionaries, f)
            log("DEBUG", "Zapisano slowniki")

    if filtered_lines:
        with open(os.path.join(opath, 'filtered_lines.pickle'), 'wb') as f:
            pickle.dump(filtered_lines, f)
            log("DEBUG", "Zapisano przefiltrowane dane")

    if char_map:
        with open(os.path.join(opath, 'char_map.pickle'), 'wb') as f:
            pickle.dump(char_map, f)
            log("DEBUG", "Zapisano mape charow")

    if char_map_inv:
        with open(os.path.join(opath, 'char_map_inv.pickle'), 'wb') as f:
            pickle.dump(char_map_inv, f)
            log("DEBUG", "Zapisano odwrocona mape charow")

    if session and sessionName:
        saver = tf.train.Saver(write_version = saver_pb2.SaverDef.V1)
        saver.save(session, os.path.join(opath, 'saved', 'model_' + sessionName + 'ckpt'))

# Iterator zbioru
def inf_train_gen(passwords = None, batch_size = None, char_map = None):
    while True:
        numpy.random.shuffle(passwords)
        for i in range(0, len(passwords) - batch_size + 1, batch_size):
            yield numpy.array([[char_map[c] for c in l] for l in passwords[i:i + batch_size]], dtype='nt32')

# Generowanie przykladow
def generate_samples(session = None, fake_inputs = None, char_map = None, char_map_inv = None):
    if session is None or fake_inputs is None or char_map is None or char_map_inv is None:
        raise ValueError('generate_samples - za malo argumentow')
    samples = numpy.argmax(session.run(fake_inputs), axis = 2)
    decoded_samples = []
    for i in range(0, len(samples)):
        decoded = []
        for j in range(0, len(samples[i])):
            decoded.append(char_map_inv[samples[i][j]])
        decoded_samples.append(tuple(decoded))
    return decoded_samples


if __name__ == '__main__':
    # Pobierz dane o dacie rozpoczecia
    start_time = datetime.datetime.fromtimestamp(time.time())
    # Wykonaj procedure trenowania
    run(parse_args())
    # Pobierz dane o dacie zakonczenia
    end_time = datetime.datetime.fromtimestamp(time.time())
    # Oblicz roznice miedzy datami: zakonczenia i rozpoczecia
    relative_delta = dateutil.relativedelta(end_time, start_time)
    # Zaloguj czas wykonywania
    log("INFO", "\nObliczenia ukonczono po: %d dniach, %d godzinach, %d minutach, %d sekundach" % (relative_delta.days, relative_delta.hours, relative_delta.minutes, relative_delta.seconds))
