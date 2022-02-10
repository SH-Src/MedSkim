import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def one_hot(size, index):
    mask = torch.Tensor(*size).fill_(0)
    if isinstance(index, Variable):
        mask = Variable(mask, volatile=index.volatile)
    ret = mask.scatter_(1, index, 1)
    return ret

class SkimRNN(nn.Module):

    def __init__(self, vocab_size, d_model, dropout, dropout_emb, a, b, c):
        super(SkimRNN, self).__init__()
        # Model hyper-parameters
        self.embed_dim = d_model
        self.emb_dropout = nn.Dropout(dropout_emb)
        self.vocab_size = vocab_size
        self.dropout = nn.Dropout(dropout)
        self.num_layers = 1
        self.large_cell_size = d_model
        self.small_cell_size = d_model // 8

        # Model modules
        self.embedding = nn.Embedding(vocab_size + 1, self.embed_dim, padding_idx=-1)
        self.large_rnn = nn.LSTMCell(input_size=self.embed_dim,
                                     hidden_size=self.large_cell_size,
                                     bias=True)
        self.small_rnn = nn.LSTMCell(input_size=self.embed_dim,
                                     hidden_size=self.small_cell_size,
                                     bias=True)

        self.linear = nn.Linear(self.embed_dim + 2 * self.large_cell_size, 2)

        self.classifier = nn.Sequential(
            nn.Linear(self.large_cell_size, 2)
        )

    def _initialize(self, batch_size, cell_size):
        init_cell = torch.Tensor(batch_size, cell_size).zero_()
        # if torch.cuda.is_available():
        #     init_cell = init_cell.cuda()
        return init_cell

    def gumbel_softmax(self, logits, temperature=1.0):
        U = torch.rand(logits.size()).to(logits.device)
        eps = 1e-20
        y = logits - torch.log(-torch.log(U + eps) + eps)
        #y = logits + self.sample_gumbel(logits.size())
        return F.softmax(y / temperature, dim=-1)

    def forward(self, x, a, b, c, d):
        """
        :param x: [batch, len]
        :return: h_state, p_list
        """
        batch_size, seq_len, num_cui_per_visit = x.size()
        x = self.embedding(x).sum(dim=2)
        embed = self.emb_dropout(x)
        #embed = self.embedding(x) # [batch, len, embed_dim]
        #batch_size = x.size()[0]

        h_state_l = self._initialize(batch_size, self.large_cell_size).to(x.device)
        h_state_s = self._initialize(batch_size, self.small_cell_size).to(x.device)
        c_l = self._initialize(batch_size, self.large_cell_size).to(x.device)
        c_s = self._initialize(batch_size, self.small_cell_size).to(x.device)

        p_ = []  # [batch, len, 2]
        h_ = []  # [batch, len, large_cell_size]

        for t in range(x.size()[1]):
            embed_ = embed[:, t, :]

            h_state_l_, c_l_ = self.large_rnn(embed_, (h_state_l, c_l))
            h_state_s, c_s = self.small_rnn(embed_, (h_state_s, c_s))

            p_t = self.linear(torch.cat([embed_.contiguous().view(-1, self.embed_dim), h_state_l_, c_l_], 1))
            r_t = self.gumbel_softmax(p_t).unsqueeze(1)

            h_state_tilde = torch.transpose(torch.stack(
                            [h_state_l_,
                             torch.cat([h_state_s[:, :self.small_cell_size],
                                        h_state_l[:, self.small_cell_size:self.large_cell_size]],
                                       dim=1)
                             ], dim=2), 1, 2)

            c_tilde = torch.transpose(torch.stack(
                            [c_l_,
                             torch.cat([c_s[:, :self.small_cell_size],
                                        c_l_[:, self.small_cell_size:self.large_cell_size]],
                                       dim=1)
                             ], dim=2), 1, 2)

            h_state_l = torch.bmm(r_t, h_state_tilde).squeeze()
            c_l = torch.bmm(r_t, c_tilde).squeeze()

            h_.append(h_state_l)
            p_.append(p_t)

        logits = self.classifier(F.relu(h_state_l))
        h_stack = torch.stack(h_, dim=1)
        p_stack = F.softmax(torch.stack(p_, dim=1), dim=-1)

        return logits


if __name__ == '__main__':
    model = SkimRNN(60, 32, 0.1, 0.1, 0, 0, 0)
    input = torch.randint(60, (16, 20, 5))
    length = torch.randint(low=1, high=20, size=(16,))
    output = model(input, 0, 0, 0, 0)
    print(output.size())
