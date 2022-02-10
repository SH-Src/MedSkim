import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LeapLSTM(nn.Module):
    def __init__(self, vocab_size, d_model, dropout, dropout_emb, a, b, c):
        super().__init__()
        self.embbedding = nn.Embedding(vocab_size + 1, d_model, padding_idx=-1)
        self.dropout = nn.Dropout(dropout)
        self.emb_dropout = nn.Dropout(dropout_emb)
        self.rnn_reverse = nn.LSTM(d_model, 10, 1, bias=False, batch_first=True)
        self.cnn = nn.Sequential(nn.Conv1d(d_model, 20, kernel_size=3, stride=1, padding=1, bias=False), nn.ReLU())
        self.leap_dense = nn.Sequential(nn.Linear(2*d_model + 30, d_model), nn.ReLU(), nn.Linear(d_model, 2))
        self.output_mlp = nn.Sequential(nn.Linear(d_model, 2))
        self.rnn_cell = nn.LSTMCell(d_model, d_model, bias=False)

    # def sample_gumbel(self, shape, eps=1e-20):
    #     U = torch.rand(shape).to(shape.device)
    #     return -torch.log(-torch.log(U + eps) + eps)

    def gumbel_softmax_sample(self, logits, temperature=1.0):
        U = torch.rand(logits.size()).to(logits.device)
        eps = 1e-20
        y = logits - torch.log(-torch.log(U + eps) + eps)
        #y = logits + self.sample_gumbel(logits.size())
        return F.softmax(y / temperature, dim=-1)

    def gumbel_softmax(self, logits, temperature=1.0, hard=False):
        """
        ST-gumple-softmax
        input: [*, n_class]
        return: flatten --> [*, n_class] an one-hot vector
        """
        y = self.gumbel_softmax_sample(logits, temperature)

        if not hard:
            return y

        shape = y.size()
        _, ind = y.max(dim=-1)
        y_hard = torch.zeros_like(y).view(-1, shape[-1])
        y_hard.scatter_(1, ind.view(-1, 1), 1)
        y_hard = y_hard.view(*shape)
        # Set gradients w.r.t. y_hard gradients w.r.t. y
        y_hard = (y_hard - y).detach() + y
        return y_hard

    def forward(self, input_seqs, mask, lengths, time_step, code_mask):
        batch_size, seq_len, num_cui_per_visit = input_seqs.size()
        x = self.embbedding(input_seqs).sum(dim=2)
        x = self.emb_dropout(x)
        rnn_reverse_output, _ = self.rnn_reverse(torch.flip(x, [1]))
        back_rnn_h = torch.flip(rnn_reverse_output, [1])
        cnn_h = self.cnn(x.permute(0, 2, 1)).permute(0, 2, 1)
        hx = x.new_zeros((batch_size, self.rnn_cell.hidden_size))
        cx = x.new_zeros((batch_size, self.rnn_cell.hidden_size))
        hiddens = []
        skips = []
        for i in range(seq_len):
            skip_prob = self.leap_dense(torch.cat((hx, x[:, i], back_rnn_h[:, i], cnn_h[:, i]), dim=1))
            skip_prob = self.gumbel_softmax(torch.log_softmax(skip_prob, dim=-1), temperature=1e-5)
            step_hx, step_cx = self.rnn_cell(x[:, i], (hx, cx))
            hx = hx.permute(1, 0) * skip_prob[:, 1] + step_hx.permute(1, 0) * skip_prob[:, 0]
            # cx = cx.permute(1, 0) * skip_prob[:, 1] + step_cx.permute(1, 0) * skip_prob[:, 0]
            hx = hx.permute(1, 0)
            # cx = cx.permute(1, 0)
            cx = step_cx
            hiddens.append(hx)
            skips.append(skip_prob[:, 1])
        hiddens = torch.stack(hiddens, dim=1)
        skips = torch.stack(skips, dim=1)
        mask = (torch.arange(seq_len, device=skips.device).unsqueeze(0).expand(batch_size, seq_len) >= lengths.unsqueeze(
            1))
        skips = skips.masked_fill(mask, 0)
        skip_rate = torch.divide(torch.sum(skips), lengths.sum())
        final_hidden = torch.gather(hiddens, 1, lengths.unsqueeze(1).unsqueeze(2).expand(batch_size, 1, self.rnn_cell.hidden_size) - 1)
        out = self.output_mlp(final_hidden).squeeze()

        return out


class LSTM_encoder2(nn.Module):
    def __init__(self, vocab_size, d_model, dropout, dropout_emb):
        super().__init__()
        self.embbedding = nn.Embedding(vocab_size + 1, d_model, padding_idx=-1)
        self.dropout = nn.Dropout(dropout)
        self.emb_dropout = nn.Dropout(dropout_emb)
        self.dropout = nn.Dropout(dropout)
        self.output_mlp = nn.Sequential(nn.Linear(d_model, 2))
        self.rnn_cell = nn.LSTMCell(d_model, d_model, bias=False)

    def forward(self, input_seqs, lengths):
        batch_size, seq_len, num_cui_per_visit = input_seqs.size()
        x = self.embbedding(input_seqs).sum(dim=2)
        x = self.emb_dropout(x)
        hx = x.new_zeros((batch_size, self.rnn_cell.hidden_size))
        cx = x.new_zeros((batch_size, self.rnn_cell.hidden_size))
        hiddens = []
        for i in range(seq_len):
            hx, cx = self.rnn_cell(x[:, i], (hx, cx))
            hiddens.append(hx)
        hiddens = torch.stack(hiddens, dim=1)
        final_hidden = torch.gather(hiddens, 1, lengths.unsqueeze(1).unsqueeze(2).expand(batch_size, 1,
                                                                                         self.rnn_cell.hidden_size) - 1)
        out = self.output_mlp(final_hidden).squeeze()
        return out

if __name__ == '__main__':
    model = LSTM_encoder(60, 32, 0.1, 0.1)
    input = torch.randint(60, (16, 20, 5))
    length = torch.randint(low=1, high=20, size=(16,))
    output, s = model(input, length)
    print(output.size())
    print(s)