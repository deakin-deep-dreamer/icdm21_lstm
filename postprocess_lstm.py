r"""Postprocess simulation module."""
import os
import sys
import traceback
import numpy as np
import pandas as pd
import argparse
import time
from datetime import datetime

import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

from data import datasource as ds
from models import commons as mcomm
from lib import post_process as post


# USER_HOME = '/home/582/ah9647'
USER_HOME = '/home/mahabib'
MODEL_PATH_STATIC = f"{USER_HOME}/py_code/models"
# MODEL_FILES = {
#     21: "pred_stream_postprocess_gru_d2_Tincartdb_100hz_1s.pt",
#     22: "pred_stream_postprocess_gru_d2_Tincartdb_100hz_2s.pt",
#     23: "pred_stream_postprocess_gru_d2_Tincartdb_100hz_3s.pt",
#     4: "pred_stream_postprocess_gru_d4_Tincartdb_100hz_2.pt",
#     8: "pred_stream_postprocess_gru_d8_Tincartdb_100hz.pt"
# }


def _find_model_file(
    lstm_depth=2, hz=100, seg_sec=1, n_blocks=None
):
    for f in os.listdir(MODEL_PATH_STATIC):
        if f.lower().endswith(".pt"):
            model_file = f"postprocess_lstm.py_PostprocessorNet_gru_d{lstm_depth}_inputSz{hz*seg_sec}_Train-incartdb_100hz_iNorm1_conv{n_blocks}_"
            if f.find(model_file) > -1:
                return f
    return None


class PostprocessorNet(nn.Module):
    r"""LSTM module to refine predicted binary stream."""

    def __init__(
        self, input_sz=None, n_layers=2, log=print, drop_prob=0.2, gru=True
    ):
        r"""Initialize LSTM module."""
        super(PostprocessorNet, self).__init__()
        self.iter = 0
        self.input_sz = input_sz
        self.n_layers = n_layers
        self.log = log
        self.gru = gru

        if self.gru:
            self.lstm_module = nn.GRU(
                input_size=input_sz, hidden_size=input_sz,
                num_layers=n_layers, batch_first=True, dropout=drop_prob
            )
        else:
            self.lstm_module = nn.LSTM(
                input_size=input_sz, hidden_size=input_sz,
                num_layers=n_layers, batch_first=True, dropout=drop_prob
            )
        r"Scoring layer"
        self.scoring_layer = nn.Conv1d(
            in_channels=1, out_channels=2, kernel_size=1, padding=0, dilation=1
        )

    def name(self):
        r"""Name current module."""
        return f"{self.__class__.__name__}_{'gru' if self.gru else 'lstm'}_d{self.n_layers}_inputSz{self.input_sz}"

    def forward(self, x, hidden):
        r"""Pass data through model."""
        self.debug(f"input: {x.shape}")

        out, hidden = self.lstm_module(x, hidden)
        self.debug(f"lstm out:{out.shape}")
        out = out.contiguous()
        self.debug(f"lstm contiguous:{out.shape}")

        out = self.scoring_layer(out)
        self.debug(f"scoring out:{out.shape}")

        self.iter += 1
        return out, hidden

    def init_hidden(self, batch_size, device):
        r"""Initialize hidden layer."""
        weight = next(self.parameters()).data
        if self.gru:
            hidden = weight.new(
                self.n_layers, batch_size, self.input_sz
            ).zero_().to(device)
        else:
            hidden = (
                weight.new(
                    self.n_layers, batch_size, self.input_sz
                ).zero_().to(device),
                weight.new(
                    self.n_layers, batch_size, self.input_sz
                ).zero_().to(device)
            )
        return hidden

    def debug(self, *args):
        r"""Output debug info."""
        if self.iter == 0:
            self.log(f"[{self.name()}], {args}")


class PredStreamDataset(Dataset):
    r"""Prediction-stream dataset."""

    def __init__(
        self, input_dir, hz=100, seg_len_sec=1, is_train=True
    ):
        r"""Construct object."""
        self.input_dir = input_dir
        self.hz = hz
        self.seg_len_sec = seg_len_sec
        self.seg_len = self.hz * self.seg_len_sec
        self.is_train = is_train

        self.record_names = []
        self.records = []
        self.record_wise_cumulative_seg_count = []
        self.initialise()

    def get_name(self):
        r"""Name the dataset, currently dataset path."""
        return self.input_dir

    def get_label_pred(self, rec_name):
        r"""Find labels and preds for given record."""
        if rec_name not in self.record_names:
            log(f"[get_record] record:{rec_name} not found.")
            return
        idx_ = self.record_names.index(rec_name)
        return self.records[idx_]

    def on_epoch_end(self):
        r"""Perform end of epoch actions."""
        pass

    def __len__(self):
        r"""Return number of total segments."""
        return self.record_wise_cumulative_seg_count[-1]

    def __getitem__(self, idx):
        r"""Return item for specified index."""
        r"Identify record from idx and then segment."
        for i, seg_count in enumerate(self.record_wise_cumulative_seg_count):
            if idx < seg_count:
                seg_offset = idx if i == 0 else idx - self.record_wise_cumulative_seg_count[i-1]
                trainY = self.records[i][0][seg_offset*self.seg_len:(seg_offset+1)*self.seg_len]
                trainX = self.records[i][1][seg_offset*self.seg_len:(seg_offset+1)*self.seg_len]
                # if seg_offset == 0:
                #     log(f"Serve segments from record:{self.record_names[i]}, X:{trainX[:30]}, Y:{trainY[:30]}")
                X_tensor = Variable(torch.from_numpy(trainX)).type(torch.FloatTensor)
                X_tensor = X_tensor.view(1, -1)
                Y_tensor = torch.from_numpy(trainY).type(torch.LongTensor)
                return X_tensor, Y_tensor
        log(f"PredStreamDataset getitem() failed to find segment, idx:{idx}")

    def initialise(self):
        r"""Load data and populate instance variables."""
        r"Exclude fold5 names for internal validation."
        cumulative_seg_count = 0
        for f in os.listdir(self.input_dir):
            g = os.path.join(self.input_dir, f)

            r"If train, ignore fold5."
            if self.is_train and f.lower().startswith("fold5"):
                continue
            elif not self.is_train and not f.lower().startswith("fold5"):
                continue

            r"3s segmentation no overlap."
            lbl_preds = pd.read_csv(g, header=None).values
            labels_, preds_ = lbl_preds.T[0], lbl_preds.T[1]
            if len(labels_) != len(preds_):
                print(f"Mismatch labels ({len(labels_)}) & preds ({len(preds_)}), ignore {f}")
                continue

            r"Calculate total seg_len_sec sec segments"
            cumulative_seg_count += (len(labels_) // (self.seg_len))
            self.record_names.append(f)
            self.records.append((labels_, preds_))
            self.record_wise_cumulative_seg_count.append(cumulative_seg_count)


def persist_vec(
    file_path, rec_name, vec, file_ext
):
    r"""Persist a vector with specified suffix."""
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    count_lines = 0
    with open(f"{file_path}/{rec_name}.{file_ext}", "w") as f:
        for d in vec:
            f.write(f"{d:.0f}\n")
            count_lines += 1
    # log(f"{count_lines} lines persisted to {file_path}/{rec_name}.{file_ext}")


def persist_label_pred(
    file_path, rec_name, labels, preds, rec_prefix=None, file_ext="pred_bin"
):
    r"""Persist predictions."""
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    count_lines = 0
    with open(f"{file_path}/{rec_prefix if rec_prefix else ''}_{rec_name}.{file_ext}", "w") as file_raw_pred:
        for _pred, _label in zip(preds, labels):
            file_raw_pred.write(f"{_label:.0f}, {_pred:.0f}\n")
            count_lines += 1
    # log(f"{count_lines} sample preds persisted to {file_path}/{rec_name}.{file_ext}")


r"LSTM hidden layer initialisation."
lstm_hidden = None
validation_model = None


def predict(
    pred_stream, hz=100, seg_len_sec=1, n_blocks=2, model_file_idx=None, log=print,
    device="cpu", gru=True, lstm_depth=2
):
    r"""Predict QRS-region binary-mask."""
    global validation_model, lstm_hidden
    r"Load model."
    if not validation_model:
        file_ = _find_model_file(
            lstm_depth=lstm_depth, seg_sec=seg_len_sec, n_blocks=n_blocks
        )
        if file_ is None:
            log(
                f"No model found for n_blocks:{n_blocks}, lstm_depth:{lstm_depth}, seg_len_sec:{seg_len_sec}"
            )
            return
        # model_file = f"{MODEL_PATH_STATIC}/{MODEL_FILES[model_file_idx]}"
        model_file = f"{MODEL_PATH_STATIC}/{file_}"
        log(
            f"loading model: {model_file}, device:{device}, "
            f"seg_len:{seg_len_sec}, lstm_depth:{lstm_depth}, n_block:{n_blocks}"
        )
        validation_model = PostprocessorNet(
            input_sz=hz*seg_len_sec, n_layers=lstm_depth, log=log, drop_prob=0
        )
        validation_model.load_state_dict(
            torch.load(
                model_file, map_location=torch.device(device)
            )
        )
    lstm_hidden = validation_model.init_hidden(batch_size=1, device=device)
    seg_len = hz*seg_len_sec
    pred_out = []
    validation_model = validation_model.to(device)
    pred_stream = np.array(pred_stream)
    for i in range(len(pred_stream)//seg_len):
        seg = pred_stream[i*seg_len:i*seg_len+seg_len]

        validation_model.eval()
        with torch.no_grad():
            X_tensor = Variable(torch.from_numpy(seg)).type(torch.FloatTensor)
            X_tensor = X_tensor.view(1, 1, X_tensor.size()[0])
            X_tensor = X_tensor.to(device)
            lstm_hidden = lstm_hidden.data if gru else tuple([e.data for e in lstm_hidden])
            preds, lstm_hidden = validation_model(X_tensor, lstm_hidden)
            preds = preds.detach().cpu().numpy()
            out_mask = preds.argmax(axis=1)
            out_mask = out_mask.flatten()
            pred_out.extend(out_mask)
    return pred_out


def _predict(
    model, data
):
    r"""Predict output using model. Internal use."""
    global lstm_hidden, MODEL
    seg_len = SEG_LEN_SEC*Hz
    pred_out = []
    model.to(DEVICE)
    for i in range(len(data)//seg_len):
        seg = data[i*seg_len:i*seg_len+seg_len]

        model.eval()
        with torch.no_grad():
            X_tensor = Variable(torch.from_numpy(seg)).type(torch.FloatTensor)
            X_tensor = X_tensor.view(1, 1, X_tensor.size()[0])
            X_tensor = X_tensor.to(DEVICE)
            lstm_hidden = lstm_hidden.data if GRU else tuple([e.data for e in lstm_hidden])
            preds, lstm_hidden = model(X_tensor, lstm_hidden)
            preds = preds.detach().cpu().numpy()
            out_mask = preds.argmax(axis=1)
            out_mask = out_mask.flatten()
            pred_out.extend(out_mask)
    return pred_out


def _run_validation(
):
    r"""Validate model."""
    # model_file = f"{MODEL_PATH}/{MODEL_FILES[23]}"
    model_file = MODEL_FILE

    model = PostprocessorNet(
        input_sz=Hz*SEG_LEN_SEC, n_layers=LSTM_LAYER, log=log, drop_prob=0
    )
    model.load_state_dict(
        torch.load(
            model_file, map_location=torch.device(DEVICE)
        )
    )
    r"LSTM hidden initialisation."
    global lstm_hidden
    lstm_hidden = model.init_hidden(batch_size=1, device=DEVICE)

    r"Validation dataset."
    input_dir = f"{PRED_PATH}/{ds.DB_NAMES[int(args.i_db)]}"
    validation_dataset = PredStreamDataset(
        input_dir=input_dir, hz=Hz, seg_len_sec=SEG_LEN_SEC, is_train=False
    )

    persist_path = f"{BASE_PATH}/preds/multi-label/gru_pred_d{LSTM_LAYER}_{SEG_LEN_SEC}s_{datetime.now():%Y%m%d%H%M%S}"

    # post.LOG = log

    r"Validate each record."
    total_stat = [[0,0,0], [0,0,0], [0,0,0]]
    # total_n_pred_locs, total_n_pred_missed, total_n_pred_extra = 0, 0, 0
    for rec_name in validation_dataset.record_names:
        labels, preds_in = validation_dataset.get_label_pred(rec_name)
        preds_out = _predict(
            model, preds_in
        )
        persist_label_pred(
            persist_path, rec_name, labels, preds_out, file_ext="pred_bin"
        )
        r"pred-out localise and score."
        pred_nodes = post.run(
            preds_out, step=0
        )
        label_nodes = post.run(
            labels, step=0
        )
        pred_locs = [node.start_loc+(node.confidence//2) for node in pred_nodes]
        label_locs = [node.start_loc+(node.confidence//2) for node in label_nodes]
        pred_locs, pred_missed, pred_extra = post.score_q_loc(
            pred_locs, label_locs
        )
        n_pred_locs, n_pred_missed, n_pred_extra = len(pred_locs), len(pred_missed), len(pred_extra)
        # total_n_pred_locs += n_pred_locs
        # total_n_pred_missed += n_pred_missed
        # total_n_pred_extra += n_pred_extra
        total_stat[0][0] += n_pred_locs
        total_stat[0][1] += n_pred_missed
        total_stat[0][2] += n_pred_extra
        ppv, se, err, f1 = post.score(
            tp=n_pred_locs-n_pred_missed, fp=n_pred_extra, fn=n_pred_missed
        )
        long_msg = f"[{rec_name}], \n\tvstep:*, total beats:{n_pred_locs}, missed:{n_pred_missed:03d}, wrong:{n_pred_extra:03d}, se:{se:.02f}, ppv:{ppv:.02f}, f1:{f1:.02f}, err:{err:.02f}, "
        # log(
        #     f"[{rec_name}], vstep:lstm, total beats:{n_pred_locs}, missed:{n_pred_missed}, wrong:{n_pred_extra}, se:{se:.02f}, ppv:{ppv:.02f}, f1:{f1:.02f}, err:{err:.02f}, "
        # )
        persist_label_pred(
            persist_path, rec_name, label_locs, pred_locs, file_ext="pred_loc"
        )

        r"pred-in localise and score."
        for vstep in range(2):
            pred_in_nodes = post.run(
                preds_in, step=vstep
            )
            pred_in_locs = [node.start_loc+(node.confidence//2) for node in pred_in_nodes]
            pred_locs, pred_missed, pred_extra = post.score_q_loc(
                pred_in_locs, label_locs
            )
            n_pred_locs, n_pred_missed, n_pred_extra = len(pred_locs), len(pred_missed), len(pred_extra)
            total_stat[vstep+1][0] += n_pred_locs
            total_stat[vstep+1][1] += n_pred_missed
            total_stat[vstep+1][2] += n_pred_extra
            ppv, se, err, f1 = post.score(
                tp=n_pred_locs-n_pred_missed, fp=n_pred_extra, fn=n_pred_missed
            )
            long_msg = f"{long_msg},\n\tvstep:{vstep}, total beats:{n_pred_locs}, missed:{n_pred_missed:03d}, wrong:{n_pred_extra:03d}, se:{se:.02f}, ppv:{ppv:.02f}, f1:{f1:.02f}, err:{err:.02f}, "
            # log(
            #     f"[{rec_name}], vstep:{vstep}, total beats:{n_pred_locs}, missed:{n_pred_missed}, wrong:{n_pred_extra}, se:{se:.02f}, ppv:{ppv:.02f}, f1:{f1:.02f}, err:{err:.02f}, "
            # )
        log(long_msg)

        break

    "Database level metric."
    for i in range(len(total_stat)):
        total_line = total_stat[i]
        ppv, se, err, f1 = post.score(
            tp=total_line[0]-total_line[1], fp=total_line[2],
            fn=total_line[1]
        )
        log(
            f"@@Summary vstep:{'lstm' if i==0 else i-1}, se:{se:.02f}, ppv:{ppv:.02f}, f1:{f1:.02f}, err:{err:.02f}"
        )


def _training(
    model=None, train_dataloader=None, validation_dataloader=None
):
    r"""Train model."""
    early_stopping = mcomm.EarlyStopping(
        patience=EARLY_STOP_PATIENCE, path=MODEL_FILE, delta=EARLY_STOP_DELTA,
        log=log, verbose=True
    )
    model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=INIT_LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode='min', factor=LR_SCHEDULER_FACTOR,
        patience=LR_SCHEDULER_PATIENCE, verbose=True
    )

    for epoch in range(1, MAX_EPOCH+1):
        train_losses = []
        since = time.time()

        r"Model training."
        model.train()
        h = model.init_hidden(batch_size=BATCH_SIZE, device=DEVICE)
        i_batch = 0
        for inputs, labels in train_dataloader:
            i_batch += 1
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            optimiser.zero_grad()
            with torch.set_grad_enabled(True):
                h = h.data if GRU else tuple([e.data for e in h])
                preds, h = model(inputs, h)
                loss = criterion(preds, labels)
                loss.backward()
                optimiser.step()
            loss_item = loss.detach().item()
            train_losses.append(loss_item)
        pass

        r"Internal validation"
        hidden_batchwise = {}
        val_losses = []
        model.eval()
        with torch.no_grad():
            for inputs, labels in validation_dataloader:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)
                '''Dynamically init LSTM hidden layers and persist batch wise,
                since, batch matters, and then re-use it.
                '''
                h = hidden_batchwise.get(inputs.size(0))
                if h is None:
                    h = model.init_hidden(
                        batch_size=inputs.size(0), device=DEVICE
                    )
                    hidden_batchwise[inputs.size(0)] = h
                h = h.data if GRU else tuple([e.data for e in h])
                preds, h = model(inputs, h)
                loss = criterion(preds, labels)
                loss_item = loss.detach().item()
                val_losses.append(loss_item)
            pass
        avg_val_loss = np.average(val_losses)
        # avg_val_loss = round(avg_val_loss, abs(EARLY_STOP_DELTA_EXP))
        avg_train_loss = np.average(train_losses)

        r"Adjust learning rate."
        scheduler.step(avg_val_loss)

        time_elapsed = time.time() - since
        log(
            f"Epoch:{epoch}, "
            f"time:{time_elapsed//60:.0f}m {time_elapsed%60:.0f}s, "
            f"train-loss:{avg_train_loss:.5f}, val-loss:{avg_val_loss:.5f}"
        )

        r"Early stopping."
        early_stopping(avg_val_loss, model)
        if early_stopping.early_stop:
            log(f"Early stopping at epoch:{epoch}")
            break
    pass
    log("Training is done.")


def _run_training(
):
    r"""Run simulation."""
    i_train_db = int(args.i_db)
    input_dir = f"{PRED_PATH}/{ds.DB_NAMES[i_train_db]}"
    log(
        f'@@   SIMULATION started. TrainDB:{ds.DB_NAMES[i_train_db]}, '
        f'batch:{BATCH_SIZE}, device:{DEVICE}, data-dir:{input_dir}  @@'
    )

    r"Training/Validation datasets."
    train_dataset = PredStreamDataset(
        input_dir=input_dir, hz=Hz, seg_len_sec=SEG_LEN_SEC, is_train=True
    )
    train_dataloader = DataLoader(
        dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=False,
        drop_last=True
    )

    validation_dataset = PredStreamDataset(
        input_dir=input_dir, hz=Hz, seg_len_sec=SEG_LEN_SEC, is_train=False
    )
    validation_dataloader = DataLoader(
        dataset=validation_dataset, batch_size=BATCH_SIZE, shuffle=False,
        drop_last=False
    )

    MODEL = PostprocessorNet(
        input_sz=Hz*SEG_LEN_SEC, n_layers=LSTM_LAYER, log=log
    )
    global MODEL_FILE
    MODEL_FILE = f'{MODEL_PATH}/{sys.argv[0]}_{MODEL.name()}_Train-{train_dataset.get_name().split("/")[-1]}_{Hz}hz_iNorm{SIG_NORM_IDX}_conv{N_BLOCKS}_{datetime.now():%Y%m%d%H%M%S}.pt'
    log(f"model file:{MODEL_FILE}")

    _training(
        model=MODEL, train_dataloader=train_dataloader,
        validation_dataloader=validation_dataloader
    )


def log(
    msg
):
    r"""Log message."""
    # tag = __name__
    # _msg = f'{datetime.now():%Y%m%d %H:%M:%S} [{tag}] {msg}'
    _msg = f'{datetime.now():%Y%m%d %H:%M:%S} [{msg}'
    print(_msg)
    f_log.write(f'{_msg}\n')
    f_log.flush()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Simulation: Postprocessing LSTM.')
    parser.add_argument(
        "--lstm_layer", required=True, help="No. of LSTM layer"
    )
    parser.add_argument(
        "--seg_len_sec", required=False, help="Segment length in sec.", default=1
    )
    parser.add_argument(
        "--block", required=False, help="No. of CNN layer used for QRS prediction.", default=8
    )
    parser.add_argument(
        "--i_cuda", required=False, help="CUDA device ID.", default=0)
    parser.add_argument(
        "--i_db", required=False, help='Database name.', default=1)

    args = parser.parse_args()
    DB_NAME = ds.DB_NAMES[int(args.i_db)]
    N_BLOCKS = int(args.block)
    LSTM_LAYER = int(args.lstm_layer)
    SIG_NORM_IDX = 1

    r"Allocate default CUDA first, else CUDA:0, else fallback to CPU."
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if DEVICE == 'cpu':
        DEVICE = torch.device(f"cuda:{args.i_cuda}" if torch.cuda.is_available() else "cpu")

    BATCH_SIZE = 64
    Hz = 100
    SEG_LEN_SEC = int(args.seg_len_sec)
    EARLY_STOP_PATIENCE = 7
    EARLY_STOP_DELTA_EXP = -4
    EARLY_STOP_DELTA = pow(10, EARLY_STOP_DELTA_EXP)
    # MAX_EPOCH = 100
    MAX_EPOCH = 200
    # INIT_LR = 0.01
    INIT_LR = 0.001
    LR_SCHEDULER_FACTOR = 0.1
    LR_SCHEDULER_PATIENCE = 5
    MODEL = None
    MODEL_FILE = None
    GRU = True

    BASE_PATH = f"{USER_HOME}/py_code"
    PRED_PATH = f"{BASE_PATH}/preds/multi-label/DilatedInception__Dep{N_BLOCKS}_lowConv24_11_CpK24_Df1_mitdb"
    LOG_PATH = f"{BASE_PATH}/logs"
    MODEL_PATH = f"{BASE_PATH}/models"
    SIM_INFO = f'block{N_BLOCKS}_seglen{Hz*SEG_LEN_SEC}_iNorm{SIG_NORM_IDX}_lstmL{LSTM_LAYER}'
    LOG_FILE = f"{LOG_PATH}/{sys.argv[0]}_traindb{DB_NAME}_{SIM_INFO}_{datetime.now():%Y%m%d%H%M%S}.log"

    f_log = open(LOG_FILE, 'a')


    try:
        _run_training()
        _run_validation()
    except RuntimeError as err:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_tb(exc_type, file=log)

        log(f'Runtime error, {err}, detail, {sys.exc_info()}')

    except:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_tb(exc_type, file=log)

        log(f'Unknown exception, {sys.exc_info()}')
    else:
        log(f'Simulation ended successfully.')
