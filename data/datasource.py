import os
import numpy as np
import pandas as pd

import torch
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset
from scipy.stats import zscore
from sklearn.preprocessing import MinMaxScaler

DEVELOPMENT = 0

if DEVELOPMENT == 0:
    'Pippin server'
    DB_BASE_PATH = '/home/mahabib/data'
elif DEVELOPMENT == 1:
    'Gadi server'
    DB_BASE_PATH = '/home/582/ah9647/data'
# elif DEVELOPMENT == 21:
#     DB_BASE_PATH = '/scratch/ez44/data'
# elif DEVELOPMENT == 22:
#     DB_BASE_PATH = '/g/data/ez44/data'

DB_NAMES = [
    'mitdb', 'incartdb', 'qtdb', 'edb', 'stdb', 'twadb', 'nstdb', 'ltstdb',
    'svdb', 'fantasia', 'cpsc19'
]


def scale(sig, i_sig_norm=1):
    r"""Scale raw signal.

    0: no Normalisation
    1: zscore
    2: MinMax scaler
    """
    if i_sig_norm == 1:
        return zscore(sig, axis=0)
    elif i_sig_norm == 2:
        scaler = MinMaxScaler()
        data = np.array(sig).reshape(-1, 1)
        return scaler.fit_transform(data).reshape(-1)
    "no normalisation"
    return sig


def data_dir(db_name=DB_NAMES[0], hz=100):
    r"""Determine data directory."""
    return f'{DB_BASE_PATH}/{db_name}/csv/{hz}hz'


class EcgDataset(Dataset):
    r"""
    Record and label are loaded. Used by train and test datasets.

    Beat annotations (https://archive.physionet.org/physiobank/annotations.shtml#aux):
        N		Normal beat (displayed as "·" by the PhysioBank ATM, LightWAVE, pschart, and psfd)
        L		Left bundle branch block beat
        R		Right bundle branch block beat
        B		Bundle branch block beat (unspecified)
        A		Atrial premature beat
        a		Aberrated atrial premature beat
        J		Nodal (junctional) premature beat
        S		Supraventricular premature or ectopic beat (atrial or nodal)
        V		Premature ventricular contraction
        r		R-on-T premature ventricular contraction
        F		Fusion of ventricular and normal beat
        e		Atrial escape beat
        j		Nodal (junctional) escape beat
        n		Supraventricular escape beat (atrial or nodal)
        E		Ventricular escape beat
        /		Paced beat
        f		Fusion of paced and normal beat
        Q		Unclassifiable beat
        ?		Beat not classified during learning
        |		Isolated QRS-like artifact

    MIT-BIH paced-beat records: 102, 104, 107
    """

    def __init__(
        self, input_directory=None, db_name=None, hz=100, seg_sec=3, n_chan=1,
        seg_slide_sec=1, q_offset_sec=0.05, record_filter=None,
        valid_beats='NLRBAJSVFREQ/', signal_norm=False,  log=None, debug=True,
        single_point_label=False, i_sig_norm=1
    ):
        r"""Construct object."""
        self.debug = debug
        self.input_directory = input_directory
        self.hz = hz
        self.db_name = db_name
        self.signal_norm = signal_norm
        self.single_point_label = single_point_label
        self.i_sig_norm = i_sig_norm

        assert self.db_name is not None

        if input_directory is None:
            self.input_directory = f'{DB_BASE_PATH}/{db_name}/csv/{hz}hz'

        self.seg_sec = seg_sec
        self.seg_sz = hz * seg_sec
        self.n_chan = n_chan
        self.seg_slide_sec = seg_slide_sec
        self.q_offset_sec = q_offset_sec
        self.record_filter = record_filter
        self.valid_beats = valid_beats
        self.log = log if log is not None else print
        self.segments = {}

        self.initialise()

    def get_name(self):
        r"""Name the dataset, currently dataset path."""
        return self.input_directory

    def on_epoch_end(self):
        r"""Perform end of epoch actions."""
        pass

    def __len__(self):
        r"""Return data length."""
        return 0

    def __getitem__(self, idx):
        r"""Return item for specified index."""
        return None, None

    def get_record(self, rec_name):
        r"""Return a tuple of signal, label, and annotation for the record."""
        id = self.record_names.index(rec_name)
        return self.recordings[id], self.labels[id], self.annotations[id]

    def initialise(self):
        r"""Initialise the object during instantiation."""
        self.log(f'Loading data from {self.input_directory} ...')
        self.header_files = []
        self.record_names = []
        for f in os.listdir(self.input_directory):
            g = os.path.join(self.input_directory, f)
            if not f.lower().startswith('.') and f.lower().endswith('ann') and os.path.isfile(g):
                self.header_files.append(g)
                self.record_names.append(
                    (lambda x: x[:x.index('.')]) (g.split('/')[-1])
                )
                'Initialise segment dict'
                self.segments[self.record_names[-1]] = []

        self.record_names.sort()
        self.header_files.sort()
        self.log(f'{self.input_directory} contains {len(self.header_files)} header files.')
        self.log(f'Records: {len(self.record_names)}, {self.record_names[:10]}')

        self.recordings = list()
        self.annotations = list()
        self.labels = list()

        for i_header, header_file in enumerate(self.header_files):

            '''Accept only filtered records, if provided.'''
            if self.record_filter is not None and True not in [header_file.find(_rec) > -1 for _rec in self.record_filter]:
                recording, annot, label = np.array([]), np.array([]), np.array([])
            else:
                if self.debug:
                    self.log(f'Process {header_file} ...')
                recording, annot, label = self.load_data(header_file)

                # '''subtract mean to remove baseline effect'''
                # recording -= np.mean(recording, dtype=np.int8)

                '''Normalisation step'''
                if self.signal_norm:
                    # recording = zscore(recording, axis=0)
                    recording = scale(recording, self.i_sig_norm)

                if self.debug:
                    self.log(f"[{self.record_names[i_header]}] n_sample:{len(recording)}, annots:{len(annot)}")

            self.recordings.append(recording)
            self.annotations.append(annot)
            self.labels.append(label)

        '''Segmentation'''
        if self.debug:
            self.log(f'Segmenting {len(self.recordings)} records ...')
        for i_rec in range(len(self.recordings)):
            # self.log(f'Segmenting {self.record_names[i_rec]} ...')
            rec = self.recordings[i_rec]
            n_samples = rec.shape[0]
            n_window = 1 + (n_samples - self.seg_sz) // (self.hz * self.seg_slide_sec)
            # self.log(f'[{self.header_files[i_rec]}] {n_window} segments out of {n_samples} samples.')

            for i_window in range(n_window):
                start = i_window * self.hz * self.seg_slide_sec
                '''
                Keep start index of segments, then using seg_sz the full
                segment can easily be formed.
                '''
                self.segments[self.record_names[i_rec]].append(start)
            # '''Check if the last segment size equals to seg_sz.'''
            # start_last_seg = self.segments[self.record_names[i_rec]][-1]
            # if start_last_seg + self.seg_sz > n_samples:
            #     self.segments[self.record_names[i_rec]].pop(-1)

    def load_data(self, header_file):
        r"""Load data from specified file path."""
        with open(header_file, 'r') as f:
            '''cpsc19 does not have header.'''
            if self.db_name != DB_NAMES[-1]:
                header = f.readlines()[1:]  # skip header
                if self.valid_beats is not None:
                    annot = self.filter_beats(
                        header_file, header, self.valid_beats.lower(), debug=False
                    )
                else:
                    annot = list(map(lambda x: int(x.split()[1]), header))

                '''Ecg-1 series'''
                recording = pd.read_csv(header_file.replace('.ann', '.csv')).values.T[self.n_chan]
            else:
                '''cpsc19'''
                annot = pd.read_csv(header_file, header=None).values.flatten()
                # header = np.array(f.readlines())
                # annot = header.flatten()
                # print(f"[{header_file}] annot: {annot}")

                '''Ecg-1 series'''
                recording = pd.read_csv(header_file.replace('.ann', '.csv')).values.T[0]

        '''Labeling'''

        labels = np.zeros(len(recording))
        "cpsc19 landmark starts from 0, others skip 1st annotation/landmark."
        i_landmark_start = 0 if self.db_name == DB_NAMES[-1] else 1

        if self.single_point_label:
            for lm in annot[i_landmark_start:]:
                if lm < len(labels):
                    labels[lm] = 1
        else:
            n_samp_q = int(self.q_offset_sec * self.hz)
            for lm in annot[i_landmark_start:]:
                i_q_start = lm - n_samp_q
                i_q_end = lm + n_samp_q + 1

                '''Ensure boundary is not exceeded.'''
                i_q_start = i_q_start if i_q_start >= 0 else 0
                i_q_end = i_q_end if i_q_end <= len(recording) else len(recording)
                for i_q in range(i_q_start, i_q_end):
                    labels[i_q] = 1

        return recording, annot, labels

    def filter_beats(self, header_file, lines, valid_beats, debug=False):
        r"""Filter out beats."""
        beat_types = {}
        annot = []
        # quote_count = 0
        for line in lines:
            tokens = line.split()
            lm, beat_type = tokens[1], tokens[2]
            if beat_type.lower() in valid_beats:
                annot.append(int(lm))
            # if beat_type.strip() == '"':
            #     quote_count += 1

            if beat_types.get(beat_type) is None:
                beat_types[beat_type] = 0
            beat_types[beat_type] += 1

        if self.debug:
            self.log(f'[Beat-type]- {header_file}: \n{beat_types}')
        return annot


class PartialDataset(EcgDataset):
    r"""Segment the parent database records."""

    def __init__(
        self, dataset, record_names=None, seg_norm=True, log=None
    ):
        r"""Construct PartialDataset object."""
        self.memory_ds = dataset
        # self.from_first = from_first
        self.seg_sz = dataset.seg_sz
        self.seg_norm = seg_norm
        self.log = log if log is not None else print

        assert record_names is not None

        self.record_names = record_names
        # self.log(f'[{self.__class__.__name__}] training records choosen using provided record names.')
        self.segments = []
        self.labels = []

        self.initialise()

    def on_epoch_end(self):
        r"""Perform epoch-end activites, for now, shuffle segments."""
        np.random.shuffle(self.indexes)

    def __len__(self):
        r"""Calculate the number of segments."""
        return len(self.indexes)

    def __getitem__(self, idx):
        r"""Return segment of specified index."""
        ID = self.indexes[idx]
        trainX = np.array(self.segments[ID])
        trainY = self.labels[ID]

        X_tensor = Variable(torch.from_numpy(trainX)).type(torch.FloatTensor)
        X_tensor = X_tensor.view(1, -1)
        # X_tensor = X_tensor.reshape(X_tensor.size()[1], -1)
        Y_tensor = torch.from_numpy(trainY).type(torch.LongTensor)
        # Y_tensor = trainY
        return X_tensor, Y_tensor

    def initialise(self):
        r"""Segment the records."""
        for i_rec, rec_name in enumerate(self.record_names):
            rec = self.memory_ds.recordings[i_rec]
            labels = self.memory_ds.labels[i_rec]

            assert len(rec) == len(labels)

            # self.log(f'Extracting {len(self.memory_ds.segments[rec_name])} segments of record: {rec_name} ...')

            for i_seg_start in self.memory_ds.segments[rec_name]:
                seg = rec[i_seg_start:i_seg_start+self.seg_sz]

                '''Check if the last segment size equals to seg_sz.'''
                if (len(seg) != self.seg_sz):
                    self.log(f'++ Segment size mismatch, expecting {self.seg_sz}, found {len(seg)}')
                    continue

                if self.seg_norm:
                    # seg = zscore(seg, axis=0)
                    seg = scale(seg, self.memory_ds.i_sig_norm)

                self.segments.append(seg)
                self.labels.append(labels[i_seg_start:i_seg_start+self.seg_sz])

        self.indexes = [i for i in range(len(self.segments))]
