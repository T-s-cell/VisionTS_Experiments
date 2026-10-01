"""Channel-independent (univariate-sample) training/eval for VisionTS LN full-shot.

Exp_Univariate subclasses the audited Exp_Long_Term_Forecast and overrides only:
  - _get_data: ETT datasets wrapped into per-channel samples so that EVERY dataset
    yields single-variable samples (paper C.1: batch size 256 with each variable an
    individual sample). Val loader is complete & deterministic
    (shuffle=False, drop_last=False) — supplementary correction of the official
    random/truncated val (e.g. illness/36 dropped 178/434 val samples per epoch).
  - vali: element-weighted MSE over the full validation set (official averaged
    per-batch means with equal weight, wrong under a partial last batch).
  - _select_optimizer: Adam over requires_grad parameters explicitly.
Train loader keeps official semantics: shuffle=True, drop_last=True.
"""
import torch
from torch import optim
from torch.utils.data import DataLoader, Dataset

from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from data_provider.data_factory import data_provider

ETT_DATA = ('ETTh1', 'ETTh2', 'ETTm1', 'ETTm2')


class Dataset_UnivariateView(Dataset):
    """Per-channel view of an ETT dataset instance: index -> (window, channel).

    Window arithmetic mirrors data_loader.Dataset_Custom.__getitem__ exactly.
    """

    def __init__(self, base):
        self.base = base
        self.channels = base.data_x.shape[-1]
        self.windows = len(base.data_x) - base.seq_len - base.pred_len + 1

    def __getattr__(self, name):
        if name == 'base':  # not yet set (e.g. during unpickling)
            raise AttributeError(name)
        return getattr(self.base, name)

    def __len__(self):
        return self.windows * self.channels

    def __getitem__(self, index):
        b = self.base
        time_index = index // self.channels
        channel_index = index % self.channels
        s_begin = time_index
        s_end = s_begin + b.seq_len
        r_begin = s_end - b.label_len
        r_end = r_begin + b.label_len + b.pred_len
        return (b.data_x[s_begin:s_end, channel_index:channel_index + 1],
                b.data_y[r_begin:r_end, channel_index:channel_index + 1],
                b.data_stamp[s_begin:s_end],
                b.data_stamp[r_begin:r_end])


class Exp_Univariate(Exp_Long_Term_Forecast):

    def _get_data(self, flag):
        data_set, _ = data_provider(self.args, flag)
        if self.args.data in ETT_DATA:
            data_set = Dataset_UnivariateView(data_set)
        if flag == 'train':
            shuffle_flag, drop_last = True, True
        else:  # val: complete & fixed (supplementary); test: official full coverage
            shuffle_flag, drop_last = False, False
        data_loader = DataLoader(
            data_set,
            batch_size=self.args.batch_size,
            shuffle=shuffle_flag,
            num_workers=self.args.num_workers,
            drop_last=drop_last)
        return data_set, data_loader

    def _select_optimizer(self):
        return optim.Adam([p for p in self.model.parameters() if p.requires_grad],
                          lr=self.args.learning_rate)

    def vali(self, vali_data, vali_loader, criterion):
        total_sse = 0.0
        total_cnt = 0
        self.model.eval()
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in vali_loader:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                diff = outputs - batch_y
                total_sse += diff.pow(2).sum().item()
                total_cnt += diff.numel()
        self.model.train()
        return total_sse / total_cnt
