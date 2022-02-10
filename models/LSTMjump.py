from operator import setitem

import torch
from torch import nn
from torch.nn import functional
from torch.autograd import Variable
from torch.distributions import Categorical
from torch.nn.utils.rnn import pack_padded_sequence


class LSTM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, categories):
        super().__init__()
        self.embed = nn.Embedding(vocab_size + 1, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size)
        self.output = nn.Linear(hidden_size, categories)
        self.nll_loss = nn.NLLLoss()
        self.hidden_size = hidden_size

    def load_pretrained_embedding(self, embedding):
        self.embed.weight = nn.Parameter(embedding)

    def forward(self, xs, lengths, t):
        batch_size, seq_len, num_cui_per_visit = xs.size()
        xs = self.embed(xs).sum(dim=2).permute(1, 0, 2)
        embs = functional.dropout(xs, 0.1)
        lengths = lengths.view(-1).tolist()
        packed_embs = pack_padded_sequence(embs, lengths)
        hs, (h, c) = self.lstm(packed_embs)
        y = functional.log_softmax(self.output(h.view(batch_size, -1)), dim=1)
        return self.nll_loss(y, t)

    def inference(self, xs, lengths):
        batch_size, seq_len, num_cui_per_visit = xs.size()
        xs = self.embed(xs).sum(dim=2).permute(1, 0, 2)
        embs = xs
        lengths = lengths.view(-1).tolist()
        packed_embs = pack_padded_sequence(embs, lengths)
        hs, (h, c) = self.lstm(packed_embs)
        return self.output(h.view(batch_size, -1)).max(dim=1)[1]


class LSTMJump(LSTM):
    def __init__(self, vocab_size, embed_size, hidden_size, categories,
                 R=20, K=40, N=5, R_test=80, N_test=8):
        super().__init__(vocab_size, embed_size, hidden_size, categories)
        self.linear = nn.Linear(hidden_size, K + 1)
        self.baseline = nn.Linear(hidden_size, 1)
        self.mse_loss = nn.MSELoss(size_average=False)
        self._R_train = R
        self._R_test = R_test
        self._N_train = N
        self._N_test = N_test

    @property
    def R(self):
        return self._R_train if self.training else self._R_test

    @property
    def N(self):
        return self._N_train if self.training else self._N_test

    def forward(self, xs, lengths, t):
        batch_size, seq_len, num_cui_per_visit = xs.size()
        max_length = seq_len
        xs = self.embed(xs).sum(dim=2).permute(1, 0, 2)
        h = Variable(xs.data.new(1, batch_size, self.hidden_size).zero_().float(), requires_grad=False)
        state = (h, h)
        embs = functional.dropout(xs, 0.1)
        rows = xs.data.new(batch_size).zero_()
        columns = xs.data.new(range(batch_size))
        log_probs = []
        baselines = []
        masks = []
        hiddens = [None] * batch_size
        last_rows = rows.clone().fill_(max_length - 1)
        for _ in range(self.N):
            for _ in range(self.R):
                feed_previous = rows >= max_length
                rows = feed_previous.long() * last_rows + (1 - feed_previous.long()) * rows
                emb = embs[rows.long(), columns.long()]
                if feed_previous.any():
                    [setitem(hiddens, i, h[:, i, :]) for i, v in enumerate(feed_previous) if v == 1]
                h, state = self.lstm(emb[None], state)
                rows = rows + 1
                if self._finish_reading(rows, max_length):
                    break
            feed_previous = rows >= max_length
            # TODO: replace where function when it is added
            rows = feed_previous.long() * last_rows + (1 - feed_previous.long()) * rows
            if feed_previous.any():
                [setitem(hiddens, i, h[:, i, :]) for i, v in enumerate(feed_previous) if v == 1]
            h, state = self.lstm(embs[rows.long(), columns.long()][None], state)
            p = functional.softmax(self.linear(h.squeeze(0)), dim=1)
            m = Categorical(p)
            jump = m.sample()
            log_prob = m.log_prob(jump)
            log_probs.append(log_prob[:, None])
            masks.append(feed_previous[:, None])
            baselines.append(self.baseline(h.squeeze(0)))
            is_stopping = (jump.data == 0).long()
            rows = is_stopping * (last_rows + 1) + (1 - is_stopping) * (rows + jump.data)
            if self._finish_reading(rows, max_length):
                break
        if any(x is None for x in hiddens):
            [setitem(hiddens, i, h[:, i, :]) for i, v in enumerate(hiddens) if v is None]
        h = torch.cat(hiddens, dim=0)
        y = functional.log_softmax(self.output(h), dim=1)
        log_prob = torch.cat(log_probs, dim=1)
        baseline = torch.cat(baselines, dim=1)
        reward = self._get_reward(y, t).expand_as(baseline)
        # filling with 0
        mask = torch.cat(masks, dim=1)
        log_prob.data.masked_fill_(mask, 0)
        baseline.data.masked_fill_(mask, 0)
        reward.data.masked_fill(mask, 0)
        return self.nll_loss(y, t) + self._reinforce(log_prob, reward, baseline) + \
            self.mse_loss(baseline, reward)

    def inference(self, xs, lengths):
        batch_size, seq_len, num_cui_per_visit = xs.size()
        max_length = seq_len
        xs = self.embed(xs).sum(dim=2).permute(1, 0, 2)
        with torch.no_grad():
            h = xs.data.new(1, batch_size, self.hidden_size).zero_().float()
        state = (h, h)
        embs = xs
        rows = xs.data.new(batch_size).zero_()
        columns = xs.data.new(range(batch_size))
        hiddens = [None] * batch_size
        last_rows = rows.clone().fill_(max_length - 1)
        for _ in range(self.N):
            for _ in range(self.R):
                feed_previous = rows >= max_length
                rows = feed_previous.long() * last_rows + (1 - feed_previous.long()) * rows
                emb = embs[rows.long(), columns.long()]
                h, state = self.lstm(emb[None], state)
                if feed_previous.any():
                    [setitem(hiddens, i, h[:, i, :]) for i, v in enumerate(feed_previous) if v == 1]
                rows = rows + 1
                if self._finish_reading(rows, max_length):
                    break
            feed_previous = (rows >= max_length)
            # TODO: replace where function when it is added
            rows = feed_previous.long() * last_rows + (1 - feed_previous.long()) * rows
            if feed_previous.any():
                [setitem(hiddens, i, h[:, i, :]) for i, v in enumerate(feed_previous) if v == 1]
            h, state = self.lstm(embs[rows.long(), columns.long()][None], state)
            p = functional.softmax(self.linear(h.squeeze(0)), dim=1)
            jump = p.max(dim=1)[1]
            is_stopping = (jump.data == 0).long()
            rows = is_stopping * (last_rows + 1) + (1 - is_stopping) * (rows + jump.data)
            if self._finish_reading(rows, max_length):
                break
        if any(x is None for x in hiddens):
            [setitem(hiddens, i, h[:, i, :]) for i, v in enumerate(hiddens) if v is None]
        h = torch.cat(hiddens, dim=0)
        return torch.softmax(self.output(h), dim=-1)

    @staticmethod
    def _finish_reading(rows, max_length):
        return (rows >= max_length).all()

    @staticmethod
    def _update_last_hidden(rows, max_length):
        return (rows >= max_length).any()

    def _reinforce(self, log_prob, reward, baseline):
        return - torch.mean((reward - baseline) * log_prob)

    def _get_reward(self, y, t):
        correct = y.data.max(dim=1)[1].eq(t.data).float()
        return Variable(correct.masked_fill_(correct == 0., -1), requires_grad=False).unsqueeze(1)


if __name__ == '__main__':
    model = LSTMJump(60, 32, 32, 2)
    input = torch.randint(60, (20, 16))
    length = torch.randint(low=1, high=20, size=(16,))
    # time_step = torch.randint(50, size=(16, 20))
    out = model.inference(input, length)
    print(out)