'''
Parameter setting rules：
View range: 0-8000Hz            Fs=16000
Window length: 0.005s           NFFT = int(Fs*0.005) = 80
                                overlap = int(Fs*0.0025) = 40
Dynamic range: 70dB             n/a
Time steps: 1000                n/a
Frequency steps: 250
Window shape: Gaussian          default window is hanning change to gaussian
'''
IDX_FREQ_I = 0
IDX_TIME_J = 1

'''
Where 1 sets a diamond morphology which implies that diagonal elements are not considered as neighbors
And 2 sets a square mask, i.e. all elements are considered neighbors.
'''
CONNECTIVITY_MASK = 2

'''
Sampling rate, related to the Nyquist conditions, which affects 
the range frequencies we can detect.
'''
# DEFAULT_FS = 1600
DEFAULT_FS = 3200

'''
Size of the FFT window, affects frequency granularity
'''
DEFAULT_WINDOW_SIZE = 4096

'''
Ratio by which each sequential window overlaps the last and the
next window. Higher overlap will allow a higher granularity of offset
matching, but potentially more fingerprints.
'''
DEFAULT_OVERLAP_RATIO = 0.5

'''
Degree to which a fingerprint can be paired with its neighbors --higher 
will cause more fingerprints, but potentially better accuracy.
'''
DEFAULT_FAN_VALUE = 15

'''
Minimum amplitude in spectrogram in order to be considered a peak.
This can be raised to reduce number of fingerprints, but can negatively
affect accuracy.
'''
DEFAULT_AMP_MIN = 10

'''
Number of cells around an amplitude peak in the spectrogram in order
for Dejavu to consider it a spectral peak. Higher values mean less
fingerprints and faster matching, but can potentially affect accuracy.
'''
PEAK_NEIGHBORHOOD_SIZE = 20

'''
Thresholds on how close or far fingerprints can be in time in order
to be paired as a fingerprint. If your max is too low, higher values of
DEFAULT_FAN_VALUE may not perform as expected.
'''
MIN_HASH_TIME_DELTA = 0
MAX_HASH_TIME_DELTA = 200

'''
If True, will sort peaks temporally for fingerprinting;
not sorting will cut down number of fingerprints, but potentially
affect performance.
'''
PEAK_SORT = False

'''
Number of bits to throw away from the front of the SHA1 hash in the
fingerprint calculation. The more you throw away, the less storage, but
potentially higher collisions and misclassifications when identifying songs.
'''
FINGERPRINT_REDUCTION = 20

'''
Sensitivity value, the higher the  value has a higher sensitivity 
and vice versa, affecting the choice of threshold value.
'''
SENSITIVITY = 0.1

'''
Since the values are small when calculated using the statistical method, 
multiplying by a larger value.
'''
MAGNIFICATION = 1000

'''
This parameter is the matrix percentage value used in the 
calculation of the threshold, which has an impact on the calculation of 
the threshold and needs to be adjusted appropriately according to the size of the data set.
'''
MASK_P = 250

'''
Whether to check the drawing
'''
PLOTS = False

'''
Blocking value
'''
CHUNK_SIZE = 1000

'''
According to the study of previous experiments, the K_VALUE value was selected as 75.
'''
K_VALUE = 75

'''
For memory not to overflow, it is wiser to split the batch calculation.
Adjustment according to physical memory size.
'''
BATCH = 50

'''
Length of sliding window.
'''
SLIDE_WINDOW_LENGTH = int(K_VALUE * 0.1)

'''
Epsilon.
'''
EPSILON = 1e-10

