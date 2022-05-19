# coding=utf-8
import cupy as cp
import math
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import scipy.io.wavfile as wav
from pydub import AudioSegment
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import (generate_binary_structure, iterate_structure, binary_erosion)
from Parameter import *

plt.rcParams['font.sans-serif'] = 'Times New Roman'

'''
You can set the memory according to the actual physical situation
mempool = cupy.get_default_memory_pool()
with cupy.cuda.Device(0):
mempool.set_limit(size=1024 ** 3 * 12)  # 12G
'''

def Fingerprint(channel_samples, Fs=DEFAULT_FS, wsize=DEFAULT_WINDOW_SIZE, wratio=DEFAULT_OVERLAP_RATIO,
                fan_value=DEFAULT_FAN_VALUE, amp_min=DEFAULT_AMP_MIN, plots=PLOTS):
    """
    Calculating audio fingerprints
    :param channel_samples:
    :param Fs:
    :param wsize:
    :param wratio:
    :param fan_value:
    :param amp_min:
    :param plots:
    :return:
    """

    if plots:
        nframes = len(channel_samples)
        print(nframes)
        time = np.arange(0, nframes) * (1.0 / Fs)
        time = np.reshape(time, [nframes, 1])
        print(time.shape)
        plt.plot(time, channel_samples, c="b")
        plt.title('%d samples' % len(channel_samples))
        plt.xlabel('time (s)')
        plt.ylabel('amplitude (A)')
        plt.show()

    overlap = int(wsize * wratio)
    arr2D = mlab.specgram(channel_samples, NFFT=wsize, Fs=Fs, window=mlab.window_hanning, noverlap=overlap)[0]

    if plots:
        plt.specgram(channel_samples, NFFT=wsize, Fs=Fs, window=mlab.window_hanning, noverlap=int(wsize * wratio),
                     mode='default', scale_by_freq=True, sides='default', scale='dB', xextent=None)
        plt.ylabel('Frequency')
        plt.xlabel('Time')
        plt.title("Spectrogram")
        plt.show()
    if plots:
        plt.plot(arr2D)
        plt.title('FFT')
        plt.show()
        plt.show()

    arr2D[arr2D == -np.inf] = 0
    arr2D = 10 * np.log10(arr2D, out=np.zeros_like(arr2D), where=(arr2D != 0))
    local_maxima = Get_2D_peaks(arr2D, plot=plots, amp_min=amp_min)
    local_maxima.sort(key=lambda x: (x[0], x[1]))
    local_maxima_row = [x[0] for x in local_maxima]
    local_maxima_column = [x[1] for x in local_maxima]
    local_maxima_freq = [x[2] for x in local_maxima]
    mask = np.zeros_like(arr2D)
    mask[local_maxima_row, local_maxima_column] = 0.5
    return mask


def Get_2D_peaks(arr2D, plot=PLOTS, amp_min=DEFAULT_AMP_MIN):
    """
    :param arr2D:
    :param plot:
    :param amp_min:
    :return:
    """
    struct = generate_binary_structure(2, CONNECTIVITY_MASK)
    # neighborhood = iterate_structure(struct, PEAK_NEIGHBORHOOD_SIZE)
    neighborhood = np.ones((PEAK_NEIGHBORHOOD_SIZE * 2 + 1, PEAK_NEIGHBORHOOD_SIZE * 2 + 1), dtype=bool)
    local_max = maximum_filter(arr2D, footprint=neighborhood) == arr2D

    background = (arr2D == 0)
    eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)

    detected_peaks = local_max != eroded_background
    amps = arr2D[detected_peaks]
    j, i = np.where(detected_peaks)

    amps = amps.flatten()
    peaks = zip(i, j, amps)
    peaks_filtered = [x for x in peaks if x[2] > amp_min]

    frequency_idx = [x[1] for x in peaks_filtered]
    time_idx = [x[0] for x in peaks_filtered]
    amps_frequency = [x[2] for x in peaks_filtered]

    font1 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 16,
             }

    if plot:
        plt.scatter(time_idx, frequency_idx, marker='o', c='red', label='Local Maxima')
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.title('Fingerprint:consists of frequency and coordinate', font1)
        plt.show()

    return list(zip(frequency_idx, time_idx, amps_frequency))


def NormMinMax(arr, min=0, max=1):
    """
     Normalization
    :param max:
    :param arr:
    :type min: object
   """
    arr_max = np.max(arr)
    arr_min = np.min(arr)
    k = (max - min) / (arr_max - arr_min + 1e-10)
    ret = min + k * (arr - arr_min)
    return ret


class Defense(object):
    def __init__(self, K, chunk_size=CHUNK_SIZE, k_value=K_VALUE, threshold=False, th='m', sets=None, th_m=0.313711, th_b=0.207398):

        self.K = K
        self.threshold = threshold
        self.sets = sets
        if self.threshold is None and self.sets is None:
            raise ValueError("Must provide explicit detection threshold or training data to calculate threshold!")
        if self.threshold is False and self.sets is not None:
            print("Explicit threshold not provided...calculating threshold for K = %d" % K)
            _, self.thresholds = Calculate_cos_similar_thresholds(self.sets, self.K)
            self.threshold = self.thresholds[-1]
        if self.threshold is True:
            if th == 'm':
                self.threshold = th_m
            if th == 'b':
                self.threshold = th_b
        print("K = %d; set threshold to: %f" % (K, self.threshold))
        self.num_queries = 0
        self.buffer = []
        self.memory = []
        self.chunk_size = chunk_size
        self.k_value = k_value
        self.history = []  # Tracks number of queries (t) when attack was detected
        self.history_by_attack = []
        self.sim = []

    def Attack_query(self, queries):

        for query in queries:
            query_fingerprint_mask = Fingerprint(query, Fs=DEFAULT_FS)
            print("query fingerprint_mask peak numbers :", np.sum(query_fingerprint_mask == 0.5))
            self.Query_into_buffer(query_fingerprint_mask)
        if len(self.buffer) <= self.K and len(self.buffer) > 0:
            last_fingerprint = self.buffer.pop(-1)
            self.num_queries -= 1
            self.Query_detected(last_fingerprint)

    def Query_into_buffer(self, query_fingerprint_mask):
        query_mask = query_fingerprint_mask
        query_mask = np.reshape(query_mask, (query_mask.shape[1], query_mask.shape[0]))
        if len(self.memory) == 0 and len(self.buffer) < self.K:
            self.buffer.append(query_mask)
            self.num_queries += 1
        else:
            self.Query_detected(query_mask)

    def Query_detected(self, query_mask):
        k = self.K
        all_similar = []
        size = query_mask.shape[0] * query_mask.shape[1]
        if len(self.buffer) > 0:  # self.buffer
            queries_mask = np.stack(self.buffer, axis=0)  # Compressed  buffer into the stack
            LAP_similar = Cos_similar(queries_mask, query_mask)
            LAP_similar = NormMinMax(LAP_similar)
            all_similar.append(LAP_similar)
        for queries_mask in self.memory:
            LAP_similar = Cos_similar(queries_mask, query_mask)
            LAP_similar = NormMinMax(LAP_similar)
            all_similar.append(LAP_similar)

        similar = np.concatenate(all_similar)
        w = W_slide_variation(similar)
        avg_sim = np.sum(similar * w)
        avg_sim = np.around(avg_sim, 6)

        self.buffer.append(query_mask)
        self.num_queries += 1

        if len(self.buffer) >= self.chunk_size:
            self.memory.append(np.stack(self.buffer, axis=0))
            self.buffer = []

        print("avg_sim: " + "%.6f" % avg_sim)
        is_attack = avg_sim > self.threshold
        if is_attack:
            self.history.append(self.num_queries)
            self.sim.append(avg_sim)
            self.Clear_memory()

    def Clear_memory(self):
        self.buffer = []
        self.memory = []

    def get_detections(self):
        history = self.history
        epochs = []
        epochs.append(history[0])
        for i in range(len(history) - 1):
            epochs.append(history[i + 1] - history[i])
        return epochs

    def get_num_detections(self):
        history = self.history
        num_detections = history[-1] / self.k_value
        return num_detections

def Calculate_thresholds(sets, K, P=MASK_P, up_to_K=False):
    """
    Calculate the threshold value
    :param sets:
    :param K:
    :param P:
    :param up_to_K:
    :return:
    """
    similarity = []
    sets_similarity = []
    for data_item in sets:
        fingerprint_mask = Fingerprint(data_item, Fs=DEFAULT_FS)
        print("fingerprint_mask numbers: ", np.sum(fingerprint_mask * 2))
        sets_similarity.append(fingerprint_mask)
    sets_similarity = np.array(sets_similarity)
    size = sets_similarity.shape[1] * sets_similarity.shape[2]
    for a in range(sets_similarity.shape[0] // P):
        mat_choose = sets_similarity[a * P:(a + 1) * P, :]
        for item in mat_choose:
            sil_mat = np.sum(np.sum((sets_similarity + item) == 1, axis=-1), axis=-1) / size
            sil_mat = -np.sort(-sil_mat, axis=-1)
            sil_mat_K = sil_mat[:K]
            sil_mat_K = NormMinMax(sil_mat_K)
            similarity.append(sil_mat_K)
    similar_matrix = np.concatenate(similarity, axis=0)

    start = 0 if up_to_K else K

    THRESHOLDS = []
    K_S = []
    for k in range(start, K + 1):
        sim_to_neighbors = similar_matrix[:k + 1]
        avg_sim_to_neighbors = sim_to_neighbors.mean(axis=-1)

        threshold = np.percentile(avg_sim_to_neighbors, SENSITIVITY)

        K_S.append(k)
        THRESHOLDS.append(threshold)

    return K_S, THRESHOLDS


def Calculate_cos_similar_thresholds(sets, K, P=MASK_P, up_to_K=False):  # sets is a 'list' include .wav filepath
    """
    Compute the threshold of similarity using cosine.
    :param sets:
    :param K:
    :param P:
    :param up_to_K:
    :return:
    """
    similarity = []
    sets_similarity = []
    for data_item in sets:
        fingerprint_mask = Fingerprint(data_item, Fs=DEFAULT_FS)
        # print("fingerprint mask peak numbers: ", np.sum(fingerprint_mask * 2))
        fingerprint_mask = np.reshape(fingerprint_mask, (fingerprint_mask.shape[1], fingerprint_mask.shape[0]))
        sets_similarity.append(fingerprint_mask)
    sets_similarity = np.array(sets_similarity)

    for a in range(sets_similarity.shape[0] // P):
        mat_choose = sets_similarity[a * P:(a + 1) * P, :]
        for item in mat_choose:
            sil_mat = Cos_similar(sets_similarity, item)
            sil_mat = -np.sort(-sil_mat, axis=-1)
            sil_mat_K = sil_mat[:K]
            similarity.append(sil_mat_K)
    similar_matrix = np.concatenate(similarity, axis=0)

    start = 0 if up_to_K else K

    THRESHOLDS = []
    K_S = []
    for k in range(start, K + 1):
        sim_to_neighbors = similar_matrix[:k + 1]
        avg_sim_to_neighbors = sim_to_neighbors.mean(axis=-1)

        threshold = np.percentile(avg_sim_to_neighbors, SENSITIVITY)

        K_S.append(k)
        THRESHOLDS.append(threshold)

    return K_S, THRESHOLDS


def Cos_similar(mat, v):
    """
    Cos_similar
    :param v
    :param mat
    :return:
    """
    batch = BATCH
    shape = mat.shape[0]
    size = mat.shape[1]
    v = cp.asarray(v)
    floor_batch = shape // batch
    if shape % batch == 0:
        batch_mat_list = [mat[t * batch:(t + 1) * batch] for t in range(0, floor_batch)]
    else:
        batch_mat_list = [mat[t * batch:(t + 1) * batch] for t in range(0, floor_batch)] + [mat[floor_batch * batch:]]
    sil_all = []
    for batch_mat in batch_mat_list:
        batch_mat = cp.asarray(batch_mat)
        mul_ab = cp.dot(batch_mat, v.T)
        norm_ab = cp.linalg.norm(cp.linalg.norm(batch_mat, axis=-1), axis=-1) * cp.linalg.norm(v)
        norm_ab = cp.reshape(norm_ab, (norm_ab.shape[0], 1, 1))
        cos_similar = mul_ab / norm_ab  # cos
        cos_similar[cp.isinf(cos_similar)] = 0.
        sil_list = cp.sum(cp.diagonal(cos_similar, axis1=-1), axis=-1) / size

        max_sil = []
        for offset in range(0, size):
            temp_sil = cp.sum(np.diagonal(cos_similar, axis1=-1, offset=offset), axis=-1) / (size - offset)
            max_sil.append(temp_sil)
        max_sil = cp.array(max_sil)
        sil_list = cp.max(max_sil, axis=0)

        sil_list = cp.asnumpy(sil_list)
        sil_all.append(sil_list)
    sil_all = np.asarray(sil_all, dtype=object)
    sil_all = np.concatenate(sil_all)
    # print("sil_all :", sil_all.shape)
    sil_all = NormMinMax(sil_all, 0, 1)

    return sil_all


def W_slide_variation(sim_list):
    """
       :param sim_list:
        :return:
    """
    max_mask_lg = SLIDE_WINDOW_LENGTH
    max_l = sim_list.shape[0]
    mask_length = 3
    all_cv = []
    for p in range(0, max_l):
        lr = mask_length // 2
        left = p - lr
        right = p + lr
        if left < 0:
            left = 0
        if right > max_l:
            right = max_l
        sub_list = sim_list[left:right + 1]
        mean, std = Mean_std(sub_list)
        cv = 1 - (std / (mean + 1e-16))
        all_cv.append(cv)
        p += 1
        if mask_length < max_mask_lg:
            mask_length += 2
    all_cv = np.array(all_cv, dtype=np.float32)
    weight = all_cv / np.sum(all_cv)
    return weight


def Mean_std(mat):
    """
       :param mat:
       :return:
    """
    return np.mean(mat), np.std(mat)


def Matching():
    pass


def np_SNR(origianl_waveform_path, target_waveform_path):
    """
         :param origianl_waveform_path
         :param target_waveform_path
         :return:
    """
    _, origianl_waveform = wav.read(origianl_waveform_path)
    _, target_waveform = wav.read(target_waveform_path)
    epsilon = EPSILON
    signal = np.sum((origianl_waveform ** 2))
    noise_per = np.sum((target_waveform - origianl_waveform) ** 2)
    print(np.sum((origianl_waveform ** 2)), signal, noise_per)
    snr = 10. * np.log10(signal / (noise_per + epsilon))
    if np.isnan(snr) or np.isinf(snr):
        snr = 1000
    return snr


def Gnoisegen(x_m, snr):
    """
         :param x_m
         :param snr
         :return:
      """
    x_m_new = []
    for x in x_m:
        Nx = x.shape[0]
        noise = np.random.randn(Nx)
        signal_power = np.sum(x * x) / Nx
        noise_power = np.sum(noise * noise) / Nx
        noise_variance = signal_power / (math.pow(10., (snr / 10)))
        noise = math.sqrt(noise_variance / noise_power) * noise
        y = x + noise
        x_m_new.append(y)
    x_m_new = np.array(x_m_new)
    x_m_new = np.reshape(x_m_new, x_m.shape)
    # print("x_m_new", x_m_new.shape)
    return x_m_new


def p_fake(audio, p_fake_audio, fake):
    """
          :param audio
          :param p_fake_audio
          :param fake:Percentage of fake inquiries.
          :return:
       """
    interval = np.floor(1 / fake)
    silce_num = int(audio.shape[0] / interval)
    perm = np.random.randint(0, p_fake_audio.shape[0], int(silce_num))
    audio_slice = [i * interval for i in range(0, silce_num)]
    audio_slice = np.array(audio_slice, dtype=np.int32)
    fake_audio = p_fake_audio[perm]
    audio[audio_slice] = fake_audio
    return audio


def Read(file_name, limit):
    """
     Data reading format processing
    :param file_name:
    :param limit:
    :return:
    """
    channels = []
    audiofile = AudioSegment.from_file(file_name, channels=1)
    data = np.frombuffer(audiofile.raw_data, np.int16)
    for chn in range(audiofile.channels):
        channels.append(data[chn::audiofile.channels])
    rate = audiofile.frame_rate
    return channels, rate


def Main(file_name: str, limit: int, print_output: bool = False):
    all_data, fs = Read(file_name, limit=0.)
    data_amount = len(all_data)
    print("all_data and fs:", data_amount, fs)

    for data_i in all_data:
        print("data_i:", len(data_i), data_i)
        Fingerprint(data_i, Fs=fs)


def getfile_path(filepath):
    """
        :param filepath
        :return:
    """
    l_dir = os.listdir(filepath)
    if str.isdigit(l_dir[0].split("_")[0]) is True:
        l_dir.sort(key=lambda x: int(re.split('_', x)[0]))
    filenames = []
    for fl in l_dir:
        full_path = os.path.join('%s/%s' % (filepath, fl))
        filenames.append(full_path)
    return filenames


def Read_File(path):
    """
    :type path: object
    """
    sets_files = getfile_path(path)
    data = []
    rates = []
    length = []
    for f in sets_files:
        wav_file, rate = Read(f, limit=0.)
        for i in range(len(wav_file)):
            data.append(wav_file[i])
            length.append(wav_file[i].shape[0])
            rates.append(rate)
    data = np.array(data, dtype=object)
    num = len(rates)
    return data, length, num


def File_Process(data, length):
    """
        :param data
        :param length
        :return:
    """
    max_length = max(length)
    min_length = min(length)
    new_data = []
    for line in data:
        l = line.shape[0]
        if l < max_length:
            new_line = np.pad(line, (0, max_length - l), 'constant', constant_values=0.)
        else:
            new_line = line[:max_length]
        new_data.append(new_line)
    new_data = np.array(new_data)
    num = new_data.shape[0]
    print("query numbers is :", num)
    print("data shape :", new_data.shape)
    return new_data


if __name__ == '__main__':
    """
    Please add the path to below code according to the actual path of AEs.
    """
    AEs, length, num = Read_File("/home/data/test")
    """
    CS-AEs-path
    """
    # AEs, length, num = Read_File("CS-AEs-audio-path")
    """
    DW-AEs-path
    """
    # AEs, length, num = Read_File("DW-AEs-audio-path")
    """
    IRTA-AEs-path
    """
    # AEs, length, num = Read_File("IRTA-AEs-audio-path")
    """
    DS-AEs-path
    """
    # AEs, length, num = Read_File(" DS-AEs-audio-path")

    AEs = File_Process(AEs, length)
    AEs = AEs + np.random.randint(-800, 800, (1300, AEs.shape[1]))
    """
    Checking the p_fake impact on detection.
    # p = ?  # Setting fake query ratios
    # p_fake_audio, _, _ = Read_File("audio-path")
    # p_fake_audio = File_Process(p_fake_audio, length)
    # AEs = p_fake(AEs, p_fake_audio, fake=p)
    """

    """
    Checking the noise impact on the detection
    # SNR = ? # Setting the S/N ratio, adding noise, according to the S/N ratio.
    # AEs = Gnoisegen(AEs, snr=SNR)
    """

    """
    Threshold is false,then calculate the threshold value.When you selected Threshold is false,you can ignore 'th'.
    Now,you need specify the 'sets' path,you can download the dataset from 
    'https://drive.google.com/file/d/1wPVK9S8TyB0aaXqXFKEebYKuKshmBvDc/view?usp=sharing'
    # sets,length,_ = Read_File("sets-audio-path")
    # sets = File_Process(sets, length)
    
    When you selected Threshold is true,you need setting 'th'.We have calculated the threshold value in advance,
    so you can opt out of the calculation.
    'th' means select 'music' or 'dialogue' as carries.
    'th=m' means select music as carry.
    'th=b' means select dialogue as carry.

    """
    # sets,length,_ = Read_File("sets-audio-path")
    # sets = File_Process(sets, length)
    detector = Defense(K=K_VALUE, sets=None, threshold=True, th='m')

    detector.Clear_memory()
    detector.Attack_query(AEs)
    detections = detector.get_detections()
    num_detections = detector.get_num_detections()
    print("AEs's number of detections: %.2f" % num_detections)
    print("AEs's cache amount per detection:", detections)
    print("AEs's which query leads to detection:", detector.history)
