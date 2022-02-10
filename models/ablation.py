import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class NoCodeSelection(nn.Module):
    def __init__(self, vocab_size, d_model, dropout, dropout_emb):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size + 1, d_model, padding_idx=-1)
        self.target_embedding = nn.Embedding(1, d_model)
        self.target_linear = nn.Linear(d_model, d_model)
        self.target_linear2 = nn.Linear(d_model, d_model)
        self.code_selection = nn.Sequential(nn.Linear(2*d_model, d_model), nn.Tanh(), nn.Linear(d_model, 2))
        self.dropout = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.emb_dropout = nn.Dropout(dropout_emb)
        self.rnn_reverse = nn.LSTM(d_model, d_model // 8, 1, bias=False, batch_first=True)
        self.visit_skip = nn.Sequential(nn.Linear(3 * d_model + d_model // 8, d_model), nn.Tanh(), nn.Linear(d_model, 2))
        self.code_query = nn.Sequential(nn.Linear(2*d_model + d_model // 8, d_model), nn.ReLU())
        # self.global_query = nn.Sequential(nn.Linear(2*d_model, d_model), nn.ReLU())
        self.output_mlp = nn.Sequential(nn.Linear(d_model, 2))
        self.layer_norm = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.fc = nn.Linear(d_model, d_model)
        self.feed_forward = nn.Sequential(nn.Linear(d_model, 4*d_model), nn.ReLU(), nn.Linear(4*d_model, d_model), nn.Dropout(dropout))
        # self.feed_forward2 = nn.Sequential(nn.Linear(d_model, 4*d_model), nn.ReLU(), nn.Linear(4*d_model, d_model))
        self.linear = nn.Linear(d_model, d_model)
        self.rnn_cell = nn.GRUCell(d_model, d_model, bias=False)
        self.time_layer = nn.Linear(1, 64)
        self.time_layer2 = nn.Linear(1, 64)
        self.time_updim = nn.Linear(64, d_model)
        self.mem_linear = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.LayerNorm(d_model))
        self.step_linear = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.LayerNorm(d_model))
        self.global_att = nn.Sequential(nn.Linear(2 * d_model + 64, d_model), nn.Tanh(), nn.Linear(d_model, 1))

    def forward(self, input_seqs, lengths, seq_time_step):
        seq_time_step = seq_time_step.unsqueeze(2) / 180
        time_feature = 1 - self.tanh(torch.pow(self.time_layer(seq_time_step), 2))
        time_encoding = self.time_updim(time_feature)
        batch_size, seq_len, num_cui_per_visit = input_seqs.size()
        x = self.embedding(input_seqs)
        x = self.emb_dropout(x)
        x = self.relu(x)
        cat_features = torch.cat((self.embedding(torch.arange(0, self.vocab_size+1).to(x.device)), self.target_embedding(x.new_zeros(self.vocab_size + 1).long())), dim=-1)
        # p = self.code_selection(cat_features)
        # p_sample = self.gumbel_softmax_sample(p)
        # p_mask = p[:, 0][input_seqs]
        x_selected = x
        selected = x_selected.sum(-2) + time_encoding
        following_feature, _ = self.rnn_reverse(selected.flip([1]))
        following_feature = torch.flip(following_feature, [1])
        hx = x.new_zeros((batch_size, self.rnn_cell.hidden_size))
        hiddens = []
        skips = []
        for i in range(seq_len):
            v_skip = self.visit_skip(torch.cat((hx, selected[:, i], following_feature[:, i], self.target_linear2(self.target_embedding(x.new_zeros(batch_size).long()))), dim=-1))
            v_skip = F.gumbel_softmax(torch.log_softmax(v_skip, dim=-1), hard=True)
            code_query = self.code_query(torch.cat((hx, following_feature[:, i], self.target_linear(self.target_embedding(x.new_zeros(batch_size).long()))), dim=-1))
            scale = np.sqrt(self.d_model)
            energy = torch.bmm(code_query.unsqueeze(1), x[:, i].permute(0, 2, 1)) / scale
            attention = torch.softmax(energy, dim=-1) # * p_mask[:, i].unsqueeze(1)
            # attention = self.gumbel_softmax_sample(attention)
            z_i = torch.matmul(attention, x[:, i]).squeeze()
            z_i = self.fc(z_i) + time_encoding[:, i]
            z_i = self.dropout2(z_i)
            # z_i = self.layer_norm(z_i)
            step_hx = self.rnn_cell(z_i, hx)
            step_hx = hx * v_skip[:, 1].unsqueeze(-1) + step_hx * v_skip[:, 0].unsqueeze(-1)
            # cx = cx * v_skip[:, 1].unsqueeze(-1) + step_cx * v_skip[:, 0].unsqueeze(-1)
            if i == 0:
                hx = step_hx
            else:
                memory = self.mem_linear(torch.stack(hiddens, dim=1))
                step_hx = self.step_linear(step_hx)
                mem_energy = torch.bmm(step_hx.unsqueeze(1), memory.permute(0, 2, 1)) / scale
                mem_attention = torch.softmax(mem_energy, dim=-1)
                hx = torch.matmul(mem_attention, memory).squeeze()
                hx = self.dropout(hx)
                hx = self.layer_norm2(hx + step_hx)
                hx = self.layer_norm(self.feed_forward(hx) + hx)
            hiddens.append(hx)
            # skips.append(v_skip[:, 1])
        hiddens = torch.stack(hiddens, dim=1)
        # skips = torch.stack(skips, dim=1)
        length_mask = (torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len) >= lengths.unsqueeze(1))
        # hiddens = hiddens.masked_fill(length_mask.unsqueeze(-1).expand_as(hiddens), float('-inf'))
        time_feature2 = 1 - self.tanh(torch.pow(self.time_layer2(seq_time_step), 2))
        global_energy = self.global_att(torch.cat((hiddens, time_feature2, self.target_embedding(x.new_zeros(batch_size, seq_len).long())), dim=-1))
        global_energy = global_energy.masked_fill(length_mask.unsqueeze(-1).expand_as(global_energy), float('-inf'))
        global_attention = torch.softmax(global_energy, dim=1)
        out_feature = global_attention * hiddens
        # final_hidden = torch.gather(hiddens, 1, lengths.unsqueeze(1).unsqueeze(2).expand(batch_size, 1, self.rnn_cell.hidden_size) - 1).squeeze()
        out = self.output_mlp(out_feature.sum(1))
        return out


class NoVSkip(nn.Module):
    def __init__(self, vocab_size, d_model, dropout, dropout_emb):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size + 1, d_model, padding_idx=-1)
        self.target_embedding = nn.Embedding(1, d_model)
        self.target_linear = nn.Linear(d_model, d_model)
        self.target_linear2 = nn.Linear(d_model, d_model)
        self.code_selection = nn.Sequential(nn.Linear(2*d_model, d_model), nn.Tanh(), nn.Linear(d_model, 2))
        self.dropout = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.emb_dropout = nn.Dropout(dropout_emb)
        self.rnn_reverse = nn.LSTM(d_model, d_model // 8, 1, bias=False, batch_first=True)
        self.visit_skip = nn.Sequential(nn.Linear(3 * d_model + d_model // 8, d_model), nn.Tanh(), nn.Linear(d_model, 2))
        self.code_query = nn.Sequential(nn.Linear(2*d_model + d_model // 8, d_model), nn.ReLU())
        # self.global_query = nn.Sequential(nn.Linear(2*d_model, d_model), nn.ReLU())
        self.output_mlp = nn.Sequential(nn.Linear(d_model, 2))
        self.layer_norm = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.fc = nn.Linear(d_model, d_model)
        self.feed_forward = nn.Sequential(nn.Linear(d_model, 4*d_model), nn.ReLU(), nn.Linear(4*d_model, d_model), nn.Dropout(dropout))
        # self.feed_forward2 = nn.Sequential(nn.Linear(d_model, 4*d_model), nn.ReLU(), nn.Linear(4*d_model, d_model))
        self.linear = nn.Linear(d_model, d_model)
        self.rnn_cell = nn.GRUCell(d_model, d_model, bias=False)
        self.time_layer = nn.Linear(1, 64)
        self.time_layer2 = nn.Linear(1, 64)
        self.time_updim = nn.Linear(64, d_model)
        self.mem_linear = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.LayerNorm(d_model))
        self.step_linear = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.LayerNorm(d_model))
        self.global_att = nn.Sequential(nn.Linear(2 * d_model + 64, d_model), nn.Tanh(), nn.Linear(d_model, 1))

    def forward(self, input_seqs, lengths, seq_time_step):
        seq_time_step = seq_time_step.unsqueeze(2) / 180
        time_feature = 1 - self.tanh(torch.pow(self.time_layer(seq_time_step), 2))
        time_encoding = self.time_updim(time_feature)
        batch_size, seq_len, num_cui_per_visit = input_seqs.size()
        x = self.embedding(input_seqs)
        x = self.emb_dropout(x)
        x = self.relu(x)
        cat_features = torch.cat((self.embedding(torch.arange(0, self.vocab_size+1).to(x.device)), self.target_embedding(x.new_zeros(self.vocab_size + 1).long())), dim=-1)
        p = self.code_selection(cat_features)
        # p_sample = self.gumbel_softmax_sample(p)
        p = F.gumbel_softmax(torch.log_softmax(p, -1), hard=True)
        p_mask = p[:, 0][input_seqs]
        x_selected = p_mask.unsqueeze(-1) * x
        selected = x_selected.sum(-2) + time_encoding
        following_feature, _ = self.rnn_reverse(selected.flip([1]))
        following_feature = torch.flip(following_feature, [1])
        hx = x.new_zeros((batch_size, self.rnn_cell.hidden_size))
        hiddens = []
        skips = []
        for i in range(seq_len):
            # v_skip = self.visit_skip(torch.cat((hx, selected[:, i], following_feature[:, i], self.target_linear2(self.target_embedding(x.new_zeros(batch_size).long()))), dim=-1))
            # v_skip = self.gumbel_softmax_sample(v_skip)
            code_query = self.code_query(torch.cat((hx, following_feature[:, i], self.target_linear(self.target_embedding(x.new_zeros(batch_size).long()))), dim=-1))
            scale = np.sqrt(self.d_model)
            energy = torch.bmm(code_query.unsqueeze(1), x[:, i].permute(0, 2, 1)) / scale
            attention = torch.softmax(energy, dim=-1) * p_mask[:, i].unsqueeze(1)
            # attention = self.gumbel_softmax_sample(attention)
            z_i = torch.matmul(attention, x[:, i]).squeeze()
            z_i = self.fc(z_i) + time_encoding[:, i]
            z_i = self.dropout2(z_i)
            # z_i = self.layer_norm(z_i)
            step_hx = self.rnn_cell(z_i, hx)
            # cx = cx * v_skip[:, 1].unsqueeze(-1) + step_cx * v_skip[:, 0].unsqueeze(-1)
            if i == 0:
                hx = step_hx
            else:
                memory = self.mem_linear(torch.stack(hiddens, dim=1))
                step_hx = self.step_linear(step_hx)
                mem_energy = torch.bmm(step_hx.unsqueeze(1), memory.permute(0, 2, 1)) / scale
                mem_attention = torch.softmax(mem_energy, dim=-1)
                hx = torch.matmul(mem_attention, memory).squeeze()
                hx = self.dropout(hx)
                hx = self.layer_norm2(hx + step_hx)
                hx = self.layer_norm(self.feed_forward(hx) + hx)
            hiddens.append(hx)
            # skips.append(v_skip[:, 1])
        hiddens = torch.stack(hiddens, dim=1)
        # skips = torch.stack(skips, dim=1)
        length_mask = (torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len) >= lengths.unsqueeze(1))
        # hiddens = hiddens.masked_fill(length_mask.unsqueeze(-1).expand_as(hiddens), float('-inf'))
        time_feature2 = 1 - self.tanh(torch.pow(self.time_layer2(seq_time_step), 2))
        global_energy = self.global_att(torch.cat((hiddens, time_feature2, self.target_embedding(x.new_zeros(batch_size, seq_len).long())), dim=-1))
        global_energy = global_energy.masked_fill(length_mask.unsqueeze(-1).expand_as(global_energy), float('-inf'))
        global_attention = torch.softmax(global_energy, dim=1)
        out_feature = global_attention * hiddens
        # final_hidden = torch.gather(hiddens, 1, lengths.unsqueeze(1).unsqueeze(2).expand(batch_size, 1, self.rnn_cell.hidden_size) - 1).squeeze()
        out = self.output_mlp(out_feature.sum(1))
        return out


class NoCSkip(nn.Module):
    def __init__(self, vocab_size, d_model, dropout, dropout_emb):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size + 1, d_model, padding_idx=-1)
        self.target_embedding = nn.Embedding(1, d_model)
        self.target_linear = nn.Linear(d_model, d_model)
        self.target_linear2 = nn.Linear(d_model, d_model)
        self.code_selection = nn.Sequential(nn.Linear(2*d_model, d_model), nn.Tanh(), nn.Linear(d_model, 2))
        self.dropout = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.emb_dropout = nn.Dropout(dropout_emb)
        self.rnn_reverse = nn.LSTM(d_model, d_model // 8, 1, bias=False, batch_first=True)
        self.visit_skip = nn.Sequential(nn.Linear(3 * d_model + d_model // 8, d_model), nn.Tanh(), nn.Linear(d_model, 2))
        self.code_query = nn.Sequential(nn.Linear(2*d_model + d_model // 8, d_model), nn.ReLU())
        # self.global_query = nn.Sequential(nn.Linear(2*d_model, d_model), nn.ReLU())
        self.output_mlp = nn.Sequential(nn.Linear(d_model, 2))
        self.layer_norm = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.fc = nn.Linear(d_model, d_model)
        self.feed_forward = nn.Sequential(nn.Linear(d_model, 4*d_model), nn.ReLU(), nn.Linear(4*d_model, d_model), nn.Dropout(dropout))
        # self.feed_forward2 = nn.Sequential(nn.Linear(d_model, 4*d_model), nn.ReLU(), nn.Linear(4*d_model, d_model))
        self.linear = nn.Linear(d_model, d_model)
        self.rnn_cell = nn.GRUCell(d_model, d_model, bias=False)
        self.time_layer = nn.Linear(1, 64)
        self.time_layer2 = nn.Linear(1, 64)
        self.time_updim = nn.Linear(64, d_model)
        self.mem_linear = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.LayerNorm(d_model))
        self.step_linear = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.LayerNorm(d_model))
        self.global_att = nn.Sequential(nn.Linear(2 * d_model + 64, d_model), nn.Tanh(), nn.Linear(d_model, 1))


    def forward(self, input_seqs, lengths, seq_time_step):
        seq_time_step = seq_time_step.unsqueeze(2) / 180
        time_feature = 1 - self.tanh(torch.pow(self.time_layer(seq_time_step), 2))
        time_encoding = self.time_updim(time_feature)
        batch_size, seq_len, num_cui_per_visit = input_seqs.size()
        x = self.embedding(input_seqs)
        x = self.emb_dropout(x)
        x = self.relu(x)
        cat_features = torch.cat((self.embedding(torch.arange(0, self.vocab_size+1).to(x.device)), self.target_embedding(x.new_zeros(self.vocab_size + 1).long())), dim=-1)
        p = self.code_selection(cat_features)
        # p_sample = self.gumbel_softmax_sample(p)
        p = F.gumbel_softmax(torch.log_softmax(p, -1), hard=True)
        p_mask = p[:, 0][input_seqs]
        x_selected = p_mask.unsqueeze(-1) * x
        selected = x_selected.sum(-2) + time_encoding
        following_feature, _ = self.rnn_reverse(selected.flip([1]))
        following_feature = torch.flip(following_feature, [1])
        hx = x.new_zeros((batch_size, self.rnn_cell.hidden_size))
        hiddens = []
        skips = []
        for i in range(seq_len):
            v_skip = self.visit_skip(torch.cat((hx, selected[:, i], following_feature[:, i], self.target_linear2(self.target_embedding(x.new_zeros(batch_size).long()))), dim=-1))
            v_skip = F.gumbel_softmax(torch.log_softmax(v_skip, dim=-1), hard=True)
            # code_query = self.code_query(torch.cat((hx, following_feature[:, i], self.target_linear(self.target_embedding(x.new_zeros(batch_size).long()))), dim=-1))
            scale = np.sqrt(self.d_model)
            # energy = torch.bmm(code_query.unsqueeze(1), x[:, i].permute(0, 2, 1)) / scale
            # attention = torch.softmax(energy, dim=-1) * p_mask[:, i].unsqueeze(1)
            # attention = self.gumbel_softmax_sample(attention)
            z_i = selected[:, i]
            z_i = self.fc(z_i) + time_encoding[:, i]
            z_i = self.dropout2(z_i)
            # z_i = self.layer_norm(z_i)
            step_hx = self.rnn_cell(z_i, hx)
            step_hx = hx * v_skip[:, 1].unsqueeze(-1) + step_hx * v_skip[:, 0].unsqueeze(-1)
            # cx = cx * v_skip[:, 1].unsqueeze(-1) + step_cx * v_skip[:, 0].unsqueeze(-1)
            if i == 0:
                hx = step_hx
            else:
                memory = self.mem_linear(torch.stack(hiddens, dim=1))
                step_hx = self.step_linear(step_hx)
                mem_energy = torch.bmm(step_hx.unsqueeze(1), memory.permute(0, 2, 1)) / scale
                mem_attention = torch.softmax(mem_energy, dim=-1)
                hx = torch.matmul(mem_attention, memory).squeeze()
                hx = self.dropout(hx)
                hx = self.layer_norm2(hx + step_hx)
                hx = self.layer_norm(self.feed_forward(hx) + hx)
            hiddens.append(hx)
            # skips.append(v_skip[:, 1])
        hiddens = torch.stack(hiddens, dim=1)
        # skips = torch.stack(skips, dim=1)
        length_mask = (torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len) >= lengths.unsqueeze(1))
        # hiddens = hiddens.masked_fill(length_mask.unsqueeze(-1).expand_as(hiddens), float('-inf'))
        time_feature2 = 1 - self.tanh(torch.pow(self.time_layer2(seq_time_step), 2))
        global_energy = self.global_att(torch.cat((hiddens, time_feature2, self.target_embedding(x.new_zeros(batch_size, seq_len).long())), dim=-1))
        global_energy = global_energy.masked_fill(length_mask.unsqueeze(-1).expand_as(global_energy), float('-inf'))
        global_attention = torch.softmax(global_energy, dim=1)
        out_feature = global_attention * hiddens
        # final_hidden = torch.gather(hiddens, 1, lengths.unsqueeze(1).unsqueeze(2).expand(batch_size, 1, self.rnn_cell.hidden_size) - 1).squeeze()
        out = self.output_mlp(out_feature.sum(1))
        return out


class NoMemory(nn.Module):
    def __init__(self, vocab_size, d_model, dropout, dropout_emb):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size + 1, d_model, padding_idx=-1)
        self.target_embedding = nn.Embedding(1, d_model)
        self.target_linear = nn.Linear(d_model, d_model)
        self.target_linear2 = nn.Linear(d_model, d_model)
        self.code_selection = nn.Sequential(nn.Linear(2*d_model, d_model), nn.Tanh(), nn.Linear(d_model, 2))
        self.dropout = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.emb_dropout = nn.Dropout(dropout_emb)
        self.rnn_reverse = nn.LSTM(d_model, d_model // 8, 1, bias=False, batch_first=True)
        self.visit_skip = nn.Sequential(nn.Linear(3 * d_model + d_model // 8, d_model), nn.Tanh(), nn.Linear(d_model, 2))
        self.code_query = nn.Sequential(nn.Linear(2*d_model + d_model // 8, d_model), nn.ReLU())
        # self.global_query = nn.Sequential(nn.Linear(2*d_model, d_model), nn.ReLU())
        self.output_mlp = nn.Sequential(nn.Linear(d_model, 2))
        self.layer_norm = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.fc = nn.Linear(d_model, d_model)
        self.feed_forward = nn.Sequential(nn.Linear(d_model, 4*d_model), nn.ReLU(), nn.Linear(4*d_model, d_model), nn.Dropout(dropout))
        # self.feed_forward2 = nn.Sequential(nn.Linear(d_model, 4*d_model), nn.ReLU(), nn.Linear(4*d_model, d_model))
        self.linear = nn.Linear(d_model, d_model)
        self.rnn_cell = nn.GRUCell(d_model, d_model, bias=False)
        self.time_layer = nn.Linear(1, 64)
        self.time_layer2 = nn.Linear(1, 64)
        self.time_updim = nn.Linear(64, d_model)
        self.mem_linear = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.LayerNorm(d_model))
        self.step_linear = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.LayerNorm(d_model))
        self.global_att = nn.Sequential(nn.Linear(2 * d_model + 64, d_model), nn.Tanh(), nn.Linear(d_model, 1))


    def forward(self, input_seqs, lengths, seq_time_step):
        seq_time_step = seq_time_step.unsqueeze(2) / 180
        time_feature = 1 - self.tanh(torch.pow(self.time_layer(seq_time_step), 2))
        time_encoding = self.time_updim(time_feature)
        batch_size, seq_len, num_cui_per_visit = input_seqs.size()
        x = self.embedding(input_seqs)
        x = self.emb_dropout(x)
        x = self.relu(x)
        cat_features = torch.cat((self.embedding(torch.arange(0, self.vocab_size+1).to(x.device)), self.target_embedding(x.new_zeros(self.vocab_size + 1).long())), dim=-1)
        p = self.code_selection(cat_features)
        # p_sample = self.gumbel_softmax_sample(p)
        p = F.gumbel_softmax(torch.log_softmax(p, -1), hard=True)
        p_mask = p[:, 0][input_seqs]
        x_selected = p_mask.unsqueeze(-1) * x
        selected = x_selected.sum(-2) + time_encoding
        following_feature, _ = self.rnn_reverse(selected.flip([1]))
        following_feature = torch.flip(following_feature, [1])
        hx = x.new_zeros((batch_size, self.rnn_cell.hidden_size))
        hiddens = []
        skips = []
        for i in range(seq_len):
            v_skip = self.visit_skip(torch.cat((hx, selected[:, i], following_feature[:, i], self.target_linear2(self.target_embedding(x.new_zeros(batch_size).long()))), dim=-1))
            v_skip = F.gumbel_softmax(torch.log_softmax(v_skip, dim=-1), hard=True)
            code_query = self.code_query(torch.cat((hx, following_feature[:, i], self.target_linear(self.target_embedding(x.new_zeros(batch_size).long()))), dim=-1))
            scale = np.sqrt(self.d_model)
            energy = torch.bmm(code_query.unsqueeze(1), x[:, i].permute(0, 2, 1)) / scale
            attention = torch.softmax(energy, dim=-1) * p_mask[:, i].unsqueeze(1)
            # attention = self.gumbel_softmax_sample(attention)
            z_i = torch.matmul(attention, x[:, i]).squeeze()
            z_i = self.fc(z_i) + time_encoding[:, i]
            z_i = self.dropout2(z_i)
            # z_i = self.layer_norm(z_i)
            step_hx = self.rnn_cell(z_i, hx)
            step_hx = hx * v_skip[:, 1].unsqueeze(-1) + step_hx * v_skip[:, 0].unsqueeze(-1)
            # cx = cx * v_skip[:, 1].unsqueeze(-1) + step_cx * v_skip[:, 0].unsqueeze(-1)
            hx = step_hx
            hiddens.append(hx)
            # skips.append(v_skip[:, 1])
        hiddens = torch.stack(hiddens, dim=1)
        # skips = torch.stack(skips, dim=1)
        length_mask = (torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len) >= lengths.unsqueeze(1))
        # hiddens = hiddens.masked_fill(length_mask.unsqueeze(-1).expand_as(hiddens), float('-inf'))
        time_feature2 = 1 - self.tanh(torch.pow(self.time_layer2(seq_time_step), 2))
        global_energy = self.global_att(torch.cat((hiddens, time_feature2, self.target_embedding(x.new_zeros(batch_size, seq_len).long())), dim=-1))
        global_energy = global_energy.masked_fill(length_mask.unsqueeze(-1).expand_as(global_energy), float('-inf'))
        global_attention = torch.softmax(global_energy, dim=1)
        out_feature = global_attention * hiddens
        # final_hidden = torch.gather(hiddens, 1, lengths.unsqueeze(1).unsqueeze(2).expand(batch_size, 1, self.rnn_cell.hidden_size) - 1).squeeze()
        out = self.output_mlp(out_feature.sum(1))
        return out


class NoTime(nn.Module):
    def __init__(self, vocab_size, d_model, dropout, dropout_emb):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size + 1, d_model, padding_idx=-1)
        self.target_embedding = nn.Embedding(1, d_model)
        self.target_linear = nn.Linear(d_model, d_model)
        self.target_linear2 = nn.Linear(d_model, d_model)
        self.code_selection = nn.Sequential(nn.Linear(2*d_model, d_model), nn.Tanh(), nn.Linear(d_model, 2))
        self.dropout = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.emb_dropout = nn.Dropout(dropout_emb)
        self.rnn_reverse = nn.LSTM(d_model, d_model // 8, 1, bias=False, batch_first=True)
        self.visit_skip = nn.Sequential(nn.Linear(3 * d_model + d_model // 8, d_model), nn.Tanh(), nn.Linear(d_model, 2))
        self.code_query = nn.Sequential(nn.Linear(2*d_model + d_model // 8, d_model), nn.ReLU())
        # self.global_query = nn.Sequential(nn.Linear(2*d_model, d_model), nn.ReLU())
        self.output_mlp = nn.Sequential(nn.Linear(d_model, 2))
        self.layer_norm = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.fc = nn.Linear(d_model, d_model)
        self.feed_forward = nn.Sequential(nn.Linear(d_model, 4*d_model), nn.ReLU(), nn.Linear(4*d_model, d_model), nn.Dropout(dropout))
        # self.feed_forward2 = nn.Sequential(nn.Linear(d_model, 4*d_model), nn.ReLU(), nn.Linear(4*d_model, d_model))
        self.linear = nn.Linear(d_model, d_model)
        self.rnn_cell = nn.GRUCell(d_model, d_model, bias=False)
        self.time_layer = nn.Linear(1, 64)
        self.time_layer2 = nn.Linear(1, 64)
        self.time_updim = nn.Linear(64, d_model)
        self.mem_linear = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.LayerNorm(d_model))
        self.step_linear = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.LayerNorm(d_model))
        self.global_att = nn.Sequential(nn.Linear(2 * d_model, d_model), nn.Tanh(), nn.Linear(d_model, 1))


    def forward(self, input_seqs, lengths, seq_time_step):
        batch_size, seq_len, num_cui_per_visit = input_seqs.size()
        x = self.embedding(input_seqs)
        x = self.emb_dropout(x)
        x = self.relu(x)
        cat_features = torch.cat((self.embedding(torch.arange(0, self.vocab_size+1).to(x.device)), self.target_embedding(x.new_zeros(self.vocab_size + 1).long())), dim=-1)
        p = self.code_selection(cat_features)
        # p_sample = self.gumbel_softmax_sample(p)
        p = F.gumbel_softmax(torch.log_softmax(p, -1), hard=True)
        p_mask = p[:, 0][input_seqs]
        x_selected = p_mask.unsqueeze(-1) * x
        selected = x_selected.sum(-2)
        following_feature, _ = self.rnn_reverse(selected.flip([1]))
        following_feature = torch.flip(following_feature, [1])
        hx = x.new_zeros((batch_size, self.rnn_cell.hidden_size))
        hiddens = []
        skips = []
        for i in range(seq_len):
            v_skip = self.visit_skip(torch.cat((hx, selected[:, i], following_feature[:, i], self.target_linear2(self.target_embedding(x.new_zeros(batch_size).long()))), dim=-1))
            # v_skip = self.gumbel_softmax_sample(v_skip)
            v_skip = F.gumbel_softmax(torch.log_softmax(v_skip, dim=-1), hard=True)
            code_query = self.code_query(torch.cat((hx, following_feature[:, i], self.target_linear(self.target_embedding(x.new_zeros(batch_size).long()))), dim=-1))
            scale = np.sqrt(self.d_model)
            energy = torch.bmm(code_query.unsqueeze(1), x[:, i].permute(0, 2, 1)) / scale
            attention = torch.softmax(energy, dim=-1) * p_mask[:, i].unsqueeze(1)
            # attention = self.gumbel_softmax_sample(attention)
            z_i = torch.matmul(attention, x[:, i]).squeeze()
            z_i = self.fc(z_i)
            z_i = self.dropout2(z_i)
            # z_i = self.layer_norm(z_i)
            step_hx = self.rnn_cell(z_i, hx)
            step_hx = hx * v_skip[:, 1].unsqueeze(-1) + step_hx * v_skip[:, 0].unsqueeze(-1)
            # cx = cx * v_skip[:, 1].unsqueeze(-1) + step_cx * v_skip[:, 0].unsqueeze(-1)
            if i == 0:
                hx = step_hx
            else:
                memory = self.mem_linear(torch.stack(hiddens, dim=1))
                step_hx = self.step_linear(step_hx)
                mem_energy = torch.bmm(step_hx.unsqueeze(1), memory.permute(0, 2, 1)) / scale
                mem_attention = torch.softmax(mem_energy, dim=-1)
                hx = torch.matmul(mem_attention, memory).squeeze()
                hx = self.dropout(hx)
                hx = self.layer_norm2(hx + step_hx)
                hx = self.layer_norm(self.feed_forward(hx) + hx)
            hiddens.append(hx)
            # skips.append(v_skip[:, 1])
        hiddens = torch.stack(hiddens, dim=1)
        # skips = torch.stack(skips, dim=1)
        length_mask = (torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len) >= lengths.unsqueeze(1))
        # hiddens = hiddens.masked_fill(length_mask.unsqueeze(-1).expand_as(hiddens), float('-inf'))
        global_energy = self.global_att(torch.cat((hiddens, self.target_embedding(x.new_zeros(batch_size, seq_len).long())), dim=-1))
        global_energy = global_energy.masked_fill(length_mask.unsqueeze(-1).expand_as(global_energy), float('-inf'))
        global_attention = torch.softmax(global_energy, dim=1)
        out_feature = global_attention * hiddens
        # final_hidden = torch.gather(hiddens, 1, lengths.unsqueeze(1).unsqueeze(2).expand(batch_size, 1, self.rnn_cell.hidden_size) - 1).squeeze()
        out = self.output_mlp(out_feature.sum(1))
        return out


class NoGlobal(nn.Module):
    def __init__(self, vocab_size, d_model, dropout, dropout_emb):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size + 1, d_model, padding_idx=-1)
        self.target_embedding = nn.Embedding(1, d_model)
        self.target_linear = nn.Linear(d_model, d_model)
        self.target_linear2 = nn.Linear(d_model, d_model)
        self.code_selection = nn.Sequential(nn.Linear(2*d_model, d_model), nn.Tanh(), nn.Linear(d_model, 2))
        self.dropout = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.emb_dropout = nn.Dropout(dropout_emb)
        self.rnn_reverse = nn.LSTM(d_model, d_model // 8, 1, bias=False, batch_first=True)
        self.visit_skip = nn.Sequential(nn.Linear(3 * d_model + d_model // 8, d_model), nn.Tanh(), nn.Linear(d_model, 2))
        self.code_query = nn.Sequential(nn.Linear(2*d_model + d_model // 8, d_model), nn.ReLU())
        # self.global_query = nn.Sequential(nn.Linear(2*d_model, d_model), nn.ReLU())
        self.output_mlp = nn.Sequential(nn.Linear(d_model, 2))
        self.layer_norm = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.fc = nn.Linear(d_model, d_model)
        self.feed_forward = nn.Sequential(nn.Linear(d_model, 4*d_model), nn.ReLU(), nn.Linear(4*d_model, d_model), nn.Dropout(dropout))
        # self.feed_forward2 = nn.Sequential(nn.Linear(d_model, 4*d_model), nn.ReLU(), nn.Linear(4*d_model, d_model))
        self.linear = nn.Linear(d_model, d_model)
        self.rnn_cell = nn.GRUCell(d_model, d_model, bias=False)
        self.time_layer = nn.Linear(1, 64)
        self.time_layer2 = nn.Linear(1, 64)
        self.time_updim = nn.Linear(64, d_model)
        self.mem_linear = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.LayerNorm(d_model))
        self.step_linear = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.LayerNorm(d_model))
        self.global_att = nn.Sequential(nn.Linear(2 * d_model + 64, d_model), nn.Tanh(), nn.Linear(d_model, 1))

    def forward(self, input_seqs, lengths, seq_time_step):
        seq_time_step = seq_time_step.unsqueeze(2) / 180
        time_feature = 1 - self.tanh(torch.pow(self.time_layer(seq_time_step), 2))
        time_encoding = self.time_updim(time_feature)
        batch_size, seq_len, num_cui_per_visit = input_seqs.size()
        x = self.embedding(input_seqs)
        x = self.emb_dropout(x)
        x = self.relu(x)
        cat_features = torch.cat((self.embedding(torch.arange(0, self.vocab_size+1).to(x.device)), self.target_embedding(x.new_zeros(self.vocab_size + 1).long())), dim=-1)
        p = self.code_selection(cat_features)
        p = F.gumbel_softmax(torch.log_softmax(p, -1), hard=True)
        p_mask = p[:, 0][input_seqs]
        x_selected = p_mask.unsqueeze(-1) * x
        selected = x_selected.sum(-2) + time_encoding
        following_feature, _ = self.rnn_reverse(selected.flip([1]))
        following_feature = torch.flip(following_feature, [1])
        hx = x.new_zeros((batch_size, self.rnn_cell.hidden_size))
        hiddens = []
        v_skips = []
        c_skips = []
        for i in range(seq_len):
            v_skip = self.visit_skip(torch.cat((hx, selected[:, i], following_feature[:, i], self.target_linear2(self.target_embedding(x.new_zeros(batch_size).long()))), dim=-1))
            v_skip = F.gumbel_softmax(torch.log_softmax(v_skip, dim=-1), hard=True)
            # print(v_skip.size())
            v_skips.append(v_skip)
            code_query = self.code_query(torch.cat((hx, following_feature[:, i], self.target_linear(self.target_embedding(x.new_zeros(batch_size).long()))), dim=-1))
            scale = np.sqrt(self.d_model)
            energy = (torch.bmm(code_query.unsqueeze(1), x[:, i].permute(0, 2, 1)) / scale)
            attention = torch.softmax(energy, dim=-1) * p_mask[:, i].unsqueeze(1)
            # print(attention.size())
            c_skips.append(attention.squeeze())
            z_i = torch.matmul(attention, x[:, i]).squeeze()
            z_i = self.fc(z_i) + time_encoding[:, i]
            z_i = self.dropout2(z_i)
            # z_i = self.layer_norm(z_i)
            step_hx = self.rnn_cell(z_i, hx)
            step_hx = hx * v_skip[:, 1].unsqueeze(-1) + step_hx * v_skip[:, 0].unsqueeze(-1)
            # cx = cx * v_skip[:, 1].unsqueeze(-1) + step_cx * v_skip[:, 0].unsqueeze(-1)
            if i == 0:
                hx = step_hx
            else:
                memory = self.mem_linear(torch.stack(hiddens, dim=1))
                step_hx = self.step_linear(step_hx)
                mem_energy = torch.bmm(step_hx.unsqueeze(1), memory.permute(0, 2, 1)) / scale
                mem_attention = torch.softmax(mem_energy, dim=-1)
                hx = torch.matmul(mem_attention, memory).squeeze()
                hx = self.dropout(hx)
                hx = self.layer_norm2(hx + step_hx)
                hx = self.layer_norm(self.feed_forward(hx) + hx)
            hiddens.append(hx)
            # skips.append(v_skip[:, 1])
        hiddens = torch.stack(hiddens, dim=1)
        v_skips = torch.stack(v_skips, dim=1)
        c_skips = torch.stack(c_skips, dim=1)
        # skips = torch.stack(skips, dim=1)
        length_mask = (torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len) >= lengths.unsqueeze(1))
        hiddens = hiddens.masked_fill(length_mask.unsqueeze(-1).expand_as(hiddens), 0)
        out_feature = hiddens
        # final_hidden = torch.gather(hiddens, 1, lengths.unsqueeze(1).unsqueeze(2).expand(batch_size, 1, self.rnn_cell.hidden_size) - 1).squeeze()
        out = self.output_mlp(out_feature.mean(1))
        return out


class NoTarget(nn.Module):
    def __init__(self, vocab_size, d_model, dropout, dropout_emb):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size + 1, d_model, padding_idx=-1)
        self.target_embedding = nn.Embedding(1, d_model)
        self.target_linear = nn.Linear(d_model, d_model)
        self.target_linear2 = nn.Linear(d_model, d_model)
        self.code_selection = nn.Sequential(nn.Linear(d_model, d_model), nn.Tanh(), nn.Linear(d_model, 2))
        self.dropout = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.emb_dropout = nn.Dropout(dropout_emb)
        self.rnn_reverse = nn.LSTM(d_model, d_model // 8, 1, bias=False, batch_first=True)
        self.visit_skip = nn.Sequential(nn.Linear(2 * d_model + d_model // 8, d_model), nn.Tanh(), nn.Linear(d_model, 2))
        self.code_query = nn.Sequential(nn.Linear(d_model + d_model // 8, d_model), nn.ReLU())
        # self.global_query = nn.Sequential(nn.Linear(2*d_model, d_model), nn.ReLU())
        self.output_mlp = nn.Sequential(nn.Linear(d_model, 2))
        self.layer_norm = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.fc = nn.Linear(d_model, d_model)
        self.feed_forward = nn.Sequential(nn.Linear(d_model, 4*d_model), nn.ReLU(), nn.Linear(4*d_model, d_model), nn.Dropout(dropout))
        # self.feed_forward2 = nn.Sequential(nn.Linear(d_model, 4*d_model), nn.ReLU(), nn.Linear(4*d_model, d_model))
        self.linear = nn.Linear(d_model, d_model)
        self.rnn_cell = nn.GRUCell(d_model, d_model, bias=False)
        self.time_layer = nn.Linear(1, 64)
        self.time_layer2 = nn.Linear(1, 64)
        self.time_updim = nn.Linear(64, d_model)
        self.mem_linear = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.LayerNorm(d_model))
        self.step_linear = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.LayerNorm(d_model))
        self.global_att = nn.Sequential(nn.Linear(d_model + 64, d_model), nn.Tanh(), nn.Linear(d_model, 1))

    def forward(self, input_seqs, lengths, seq_time_step):
        seq_time_step = seq_time_step.unsqueeze(2) / 180
        time_feature = 1 - self.tanh(torch.pow(self.time_layer(seq_time_step), 2))
        time_encoding = self.time_updim(time_feature)
        batch_size, seq_len, num_cui_per_visit = input_seqs.size()
        x = self.embedding(input_seqs)
        x = self.emb_dropout(x)
        x = self.relu(x)
        cat_features = self.embedding(torch.arange(0, self.vocab_size+1).to(x.device))
        p = self.code_selection(cat_features)
        p = F.gumbel_softmax(torch.log_softmax(p, -1), hard=True)
        p_mask = p[:, 0][input_seqs]
        x_selected = p_mask.unsqueeze(-1) * x
        selected = x_selected.sum(-2) + time_encoding
        following_feature, _ = self.rnn_reverse(selected.flip([1]))
        following_feature = torch.flip(following_feature, [1])
        hx = x.new_zeros((batch_size, self.rnn_cell.hidden_size))
        hiddens = []
        v_skips = []
        c_skips = []
        for i in range(seq_len):
            v_skip = self.visit_skip(torch.cat((hx, selected[:, i], following_feature[:, i]), dim=-1))
            v_skip = F.gumbel_softmax(torch.log_softmax(v_skip, dim=-1), hard=True)
            # print(v_skip.size())
            v_skips.append(v_skip)
            code_query = self.code_query(torch.cat((hx, following_feature[:, i]), dim=-1))
            scale = np.sqrt(self.d_model)
            energy = (torch.bmm(code_query.unsqueeze(1), x[:, i].permute(0, 2, 1)) / scale)
            attention = torch.softmax(energy, dim=-1) * p_mask[:, i].unsqueeze(1)
            # print(attention.size())
            c_skips.append(attention.squeeze())
            z_i = torch.matmul(attention, x[:, i]).squeeze()
            z_i = self.fc(z_i) + time_encoding[:, i]
            z_i = self.dropout2(z_i)
            # z_i = self.layer_norm(z_i)
            step_hx = self.rnn_cell(z_i, hx)
            step_hx = hx * v_skip[:, 1].unsqueeze(-1) + step_hx * v_skip[:, 0].unsqueeze(-1)
            # cx = cx * v_skip[:, 1].unsqueeze(-1) + step_cx * v_skip[:, 0].unsqueeze(-1)
            if i == 0:
                hx = step_hx
            else:
                memory = self.mem_linear(torch.stack(hiddens, dim=1))
                step_hx = self.step_linear(step_hx)
                mem_energy = torch.bmm(step_hx.unsqueeze(1), memory.permute(0, 2, 1)) / scale
                mem_attention = torch.softmax(mem_energy, dim=-1)
                hx = torch.matmul(mem_attention, memory).squeeze()
                hx = self.dropout(hx)
                hx = self.layer_norm2(hx + step_hx)
                hx = self.layer_norm(self.feed_forward(hx) + hx)
            hiddens.append(hx)
            # skips.append(v_skip[:, 1])
        hiddens = torch.stack(hiddens, dim=1)
        # v_skips = torch.stack(v_skips, dim=1)
        # c_skips = torch.stack(c_skips, dim=1)
        # one = torch.ones_like(c_skips)
        # c_skips = torch.where(c_skips > 0, one, c_skips)
        # code_mask = torch.where(code_mask > 0, one, code_mask)
        # skips = v_skips[:, :, 0].unsqueeze(-1) * c_skips
        # skip_rate = 1 - torch.divide(torch.sum(skips), torch.sum(1 - code_mask))
        length_mask = (torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len) >= lengths.unsqueeze(1))
        # hiddens = hiddens.masked_fill(length_mask.unsqueeze(-1).expand_as(hiddens), float('-inf'))
        time_feature2 = 1 - self.tanh(torch.pow(self.time_layer2(seq_time_step), 2))
        global_energy = self.global_att(torch.cat((hiddens, time_feature2), dim=-1))
        global_energy = global_energy.masked_fill(length_mask.unsqueeze(-1).expand_as(global_energy), float('-inf'))
        global_attention = torch.softmax(global_energy, dim=1)
        out_feature = global_attention * hiddens
        # final_hidden = torch.gather(hiddens, 1, lengths.unsqueeze(1).unsqueeze(2).expand(batch_size, 1, self.rnn_cell.hidden_size) - 1).squeeze()
        out = self.output_mlp(out_feature.sum(1))
        return out


class NoFollowing(nn.Module):
    def __init__(self, vocab_size, d_model, dropout, dropout_emb):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size + 1, d_model, padding_idx=-1)
        self.target_embedding = nn.Embedding(1, d_model)
        self.target_linear = nn.Linear(d_model, d_model)
        self.target_linear2 = nn.Linear(d_model, d_model)
        self.code_selection = nn.Sequential(nn.Linear(2*d_model, d_model), nn.Tanh(), nn.Linear(d_model, 2))
        self.dropout = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.emb_dropout = nn.Dropout(dropout_emb)
        self.rnn_reverse = nn.LSTM(d_model, d_model // 8, 1, bias=False, batch_first=True)
        self.visit_skip = nn.Sequential(nn.Linear(3 * d_model, d_model), nn.Tanh(), nn.Linear(d_model, 2))
        self.code_query = nn.Sequential(nn.Linear(2*d_model, d_model), nn.ReLU())
        # self.global_query = nn.Sequential(nn.Linear(2*d_model, d_model), nn.ReLU())
        self.output_mlp = nn.Sequential(nn.Linear(d_model, 2))
        self.layer_norm = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.fc = nn.Linear(d_model, d_model)
        self.feed_forward = nn.Sequential(nn.Linear(d_model, 4*d_model), nn.ReLU(), nn.Linear(4*d_model, d_model), nn.Dropout(dropout))
        # self.feed_forward2 = nn.Sequential(nn.Linear(d_model, 4*d_model), nn.ReLU(), nn.Linear(4*d_model, d_model))
        self.linear = nn.Linear(d_model, d_model)
        self.rnn_cell = nn.GRUCell(d_model, d_model, bias=False)
        self.time_layer = nn.Linear(1, 64)
        self.time_layer2 = nn.Linear(1, 64)
        self.time_updim = nn.Linear(64, d_model)
        self.mem_linear = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.LayerNorm(d_model))
        self.step_linear = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.LayerNorm(d_model))
        self.global_att = nn.Sequential(nn.Linear(2 * d_model + 64, d_model), nn.Tanh(), nn.Linear(d_model, 1))


    def forward(self, input_seqs, lengths, seq_time_step):
        seq_time_step = seq_time_step.unsqueeze(2) / 180
        time_feature = 1 - self.tanh(torch.pow(self.time_layer(seq_time_step), 2))
        time_encoding = self.time_updim(time_feature)
        batch_size, seq_len, num_cui_per_visit = input_seqs.size()
        x = self.embedding(input_seqs)
        x = self.emb_dropout(x)
        x = self.relu(x)
        cat_features = torch.cat((self.embedding(torch.arange(0, self.vocab_size+1).to(x.device)), self.target_embedding(x.new_zeros(self.vocab_size + 1).long())), dim=-1)
        p = self.code_selection(cat_features)
        p = F.gumbel_softmax(torch.log_softmax(p, -1), hard=True)
        p_mask = p[:, 0][input_seqs]
        x_selected = p_mask.unsqueeze(-1) * x
        selected = x_selected.sum(-2) + time_encoding
        # following_feature, _ = self.rnn_reverse(selected.flip([1]))
        # following_feature = torch.flip(following_feature, [1])
        hx = x.new_zeros((batch_size, self.rnn_cell.hidden_size))
        hiddens = []
        v_skips = []
        c_skips = []
        for i in range(seq_len):
            v_skip = self.visit_skip(torch.cat((hx, selected[:, i], self.target_linear2(self.target_embedding(x.new_zeros(batch_size).long()))), dim=-1))
            v_skip = F.gumbel_softmax(torch.log_softmax(v_skip, dim=-1), hard=True)
            # print(v_skip.size())
            v_skips.append(v_skip)
            code_query = self.code_query(torch.cat((hx, self.target_linear(self.target_embedding(x.new_zeros(batch_size).long()))), dim=-1))
            scale = np.sqrt(self.d_model)
            energy = (torch.bmm(code_query.unsqueeze(1), x[:, i].permute(0, 2, 1)) / scale)
            attention = torch.softmax(energy, dim=-1) * p_mask[:, i].unsqueeze(1)
            # print(attention.size())
            c_skips.append(attention.squeeze())
            z_i = torch.matmul(attention, x[:, i]).squeeze()
            z_i = self.fc(z_i) + time_encoding[:, i]
            z_i = self.dropout2(z_i)
            # z_i = self.layer_norm(z_i)
            step_hx = self.rnn_cell(z_i, hx)
            step_hx = hx * v_skip[:, 1].unsqueeze(-1) + step_hx * v_skip[:, 0].unsqueeze(-1)
            # cx = cx * v_skip[:, 1].unsqueeze(-1) + step_cx * v_skip[:, 0].unsqueeze(-1)
            if i == 0:
                hx = step_hx
            else:
                memory = self.mem_linear(torch.stack(hiddens, dim=1))
                step_hx = self.step_linear(step_hx)
                mem_energy = torch.bmm(step_hx.unsqueeze(1), memory.permute(0, 2, 1)) / scale
                mem_attention = torch.softmax(mem_energy, dim=-1)
                hx = torch.matmul(mem_attention, memory).squeeze()
                hx = self.dropout(hx)
                hx = self.layer_norm2(hx + step_hx)
                hx = self.layer_norm(self.feed_forward(hx) + hx)
            hiddens.append(hx)
            # skips.append(v_skip[:, 1])
        hiddens = torch.stack(hiddens, dim=1)
        v_skips = torch.stack(v_skips, dim=1)
        c_skips = torch.stack(c_skips, dim=1)
        one = torch.ones_like(c_skips)
        c_skips = torch.where(c_skips > 0, one, c_skips)
        # code_mask = torch.where(code_mask > 0, one, code_mask)
        # skips = v_skips[:, :, 0].unsqueeze(-1) * c_skips
        # skip_rate = 1 - torch.divide(torch.sum(skips), torch.sum(1 - code_mask))
        length_mask = (torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len) >= lengths.unsqueeze(1))
        # hiddens = hiddens.masked_fill(length_mask.unsqueeze(-1).expand_as(hiddens), float('-inf'))
        time_feature2 = 1 - self.tanh(torch.pow(self.time_layer2(seq_time_step), 2))
        global_energy = self.global_att(torch.cat((hiddens, time_feature2, self.target_embedding(x.new_zeros(batch_size, seq_len).long())), dim=-1))
        global_energy = global_energy.masked_fill(length_mask.unsqueeze(-1).expand_as(global_energy), float('-inf'))
        global_attention = torch.softmax(global_energy, dim=1)
        out_feature = global_attention * hiddens
        # final_hidden = torch.gather(hiddens, 1, lengths.unsqueeze(1).unsqueeze(2).expand(batch_size, 1, self.rnn_cell.hidden_size) - 1).squeeze()
        out = self.output_mlp(out_feature.sum(1))
        return out


class NoGumbel(nn.Module):
    def __init__(self, vocab_size, d_model, dropout, dropout_emb):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size + 1, d_model, padding_idx=-1)
        self.target_embedding = nn.Embedding(1, d_model)
        self.target_linear = nn.Linear(d_model, d_model)
        self.target_linear2 = nn.Linear(d_model, d_model)
        self.code_selection = nn.Sequential(nn.Linear(2*d_model, d_model), nn.Tanh(), nn.Linear(d_model, 2))
        self.dropout = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.emb_dropout = nn.Dropout(dropout_emb)
        self.rnn_reverse = nn.LSTM(d_model, d_model // 8, 1, bias=False, batch_first=True)
        self.visit_skip = nn.Sequential(nn.Linear(3 * d_model + d_model // 8, d_model), nn.Tanh(), nn.Linear(d_model, 2))
        self.code_query = nn.Sequential(nn.Linear(2*d_model + d_model // 8, d_model), nn.ReLU())
        # self.global_query = nn.Sequential(nn.Linear(2*d_model, d_model), nn.ReLU())
        self.output_mlp = nn.Sequential(nn.Linear(d_model, 2))
        self.layer_norm = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.fc = nn.Linear(d_model, d_model)
        self.feed_forward = nn.Sequential(nn.Linear(d_model, 4*d_model), nn.ReLU(), nn.Linear(4*d_model, d_model), nn.Dropout(dropout))
        # self.feed_forward2 = nn.Sequential(nn.Linear(d_model, 4*d_model), nn.ReLU(), nn.Linear(4*d_model, d_model))
        self.linear = nn.Linear(d_model, d_model)
        self.rnn_cell = nn.GRUCell(d_model, d_model, bias=False)
        self.time_layer = nn.Linear(1, 64)
        self.time_layer2 = nn.Linear(1, 64)
        self.time_updim = nn.Linear(64, d_model)
        self.mem_linear = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.LayerNorm(d_model))
        self.step_linear = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.LayerNorm(d_model))
        self.global_att = nn.Sequential(nn.Linear(2 * d_model + 64, d_model), nn.Tanh(), nn.Linear(d_model, 1))


    def forward(self, input_seqs, lengths, seq_time_step):
        seq_time_step = seq_time_step.unsqueeze(2) / 180
        time_feature = 1 - self.tanh(torch.pow(self.time_layer(seq_time_step), 2))
        time_encoding = self.time_updim(time_feature)
        batch_size, seq_len, num_cui_per_visit = input_seqs.size()
        x = self.embedding(input_seqs)
        x = self.emb_dropout(x)
        x = self.relu(x)
        cat_features = torch.cat((self.embedding(torch.arange(0, self.vocab_size+1).to(x.device)), self.target_embedding(x.new_zeros(self.vocab_size + 1).long())), dim=-1)
        p = self.code_selection(cat_features)
        p = torch.softmax(p, -1)
        p_mask = p[:, 0][input_seqs]
        x_selected = p_mask.unsqueeze(-1) * x
        selected = x_selected.sum(-2) + time_encoding
        following_feature, _ = self.rnn_reverse(selected.flip([1]))
        following_feature = torch.flip(following_feature, [1])
        hx = x.new_zeros((batch_size, self.rnn_cell.hidden_size))
        hiddens = []
        v_skips = []
        c_skips = []
        for i in range(seq_len):
            v_skip = self.visit_skip(torch.cat((hx, selected[:, i], following_feature[:, i], self.target_linear2(self.target_embedding(x.new_zeros(batch_size).long()))), dim=-1))
            v_skip = torch.softmax(v_skip, dim=-1)
            # print(v_skip.size())
            v_skips.append(v_skip)
            code_query = self.code_query(torch.cat((hx, following_feature[:, i], self.target_linear(self.target_embedding(x.new_zeros(batch_size).long()))), dim=-1))
            scale = np.sqrt(self.d_model)
            energy = (torch.bmm(code_query.unsqueeze(1), x[:, i].permute(0, 2, 1)) / scale)
            attention = torch.softmax(energy, dim=-1) * p_mask[:, i].unsqueeze(1)
            # print(attention.size())
            c_skips.append(attention.squeeze())
            z_i = torch.matmul(attention, x[:, i]).squeeze()
            z_i = self.fc(z_i) + time_encoding[:, i]
            z_i = self.dropout2(z_i)
            # z_i = self.layer_norm(z_i)
            step_hx = self.rnn_cell(z_i, hx)
            step_hx = hx * v_skip[:, 1].unsqueeze(-1) + step_hx * v_skip[:, 0].unsqueeze(-1)
            # cx = cx * v_skip[:, 1].unsqueeze(-1) + step_cx * v_skip[:, 0].unsqueeze(-1)
            if i == 0:
                hx = step_hx
            else:
                memory = self.mem_linear(torch.stack(hiddens, dim=1))
                step_hx = self.step_linear(step_hx)
                mem_energy = torch.bmm(step_hx.unsqueeze(1), memory.permute(0, 2, 1)) / scale
                mem_attention = torch.softmax(mem_energy, dim=-1)
                hx = torch.matmul(mem_attention, memory).squeeze()
                hx = self.dropout(hx)
                hx = self.layer_norm2(hx + step_hx)
                hx = self.layer_norm(self.feed_forward(hx) + hx)
            hiddens.append(hx)
            # skips.append(v_skip[:, 1])
        hiddens = torch.stack(hiddens, dim=1)
        # v_skips = torch.stack(v_skips, dim=1)
        # c_skips = torch.stack(c_skips, dim=1)
        # one = torch.ones_like(c_skips)
        # c_skips = torch.where(c_skips > 0, one, c_skips)
        # code_mask = torch.where(code_mask > 0, one, code_mask)
        # skips = v_skips[:, :, 0].unsqueeze(-1) * c_skips
        # skip_rate = 1 - torch.divide(torch.sum(skips), torch.sum(1 - code_mask))
        length_mask = (torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len) >= lengths.unsqueeze(1))
        # hiddens = hiddens.masked_fill(length_mask.unsqueeze(-1).expand_as(hiddens), float('-inf'))
        time_feature2 = 1 - self.tanh(torch.pow(self.time_layer2(seq_time_step), 2))
        global_energy = self.global_att(torch.cat((hiddens, time_feature2, self.target_embedding(x.new_zeros(batch_size, seq_len).long())), dim=-1))
        global_energy = global_energy.masked_fill(length_mask.unsqueeze(-1).expand_as(global_energy), float('-inf'))
        global_attention = torch.softmax(global_energy, dim=1)
        out_feature = global_attention * hiddens
        # final_hidden = torch.gather(hiddens, 1, lengths.unsqueeze(1).unsqueeze(2).expand(batch_size, 1, self.rnn_cell.hidden_size) - 1).squeeze()
        out = self.output_mlp(out_feature.sum(1))
        return out


class NoPenalty(nn.Module):
    def __init__(self, vocab_size, d_model, dropout, dropout_emb):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size + 1, d_model, padding_idx=-1)
        self.target_embedding = nn.Embedding(1, d_model)
        self.target_linear = nn.Linear(d_model, d_model)
        self.target_linear2 = nn.Linear(d_model, d_model)
        self.code_selection = nn.Sequential(nn.Linear(2*d_model, d_model), nn.Tanh(), nn.Linear(d_model, 2))
        self.dropout = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.emb_dropout = nn.Dropout(dropout_emb)
        self.rnn_reverse = nn.LSTM(d_model, d_model // 8, 1, bias=False, batch_first=True)
        self.visit_skip = nn.Sequential(nn.Linear(3 * d_model + d_model // 8, d_model), nn.Tanh(), nn.Linear(d_model, 2))
        self.code_query = nn.Sequential(nn.Linear(2*d_model + d_model // 8, d_model), nn.ReLU())
        # self.global_query = nn.Sequential(nn.Linear(2*d_model, d_model), nn.ReLU())
        self.output_mlp = nn.Sequential(nn.Linear(d_model, 2))
        self.layer_norm = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.fc = nn.Linear(d_model, d_model)
        self.feed_forward = nn.Sequential(nn.Linear(d_model, 4*d_model), nn.ReLU(), nn.Linear(4*d_model, d_model), nn.Dropout(dropout))
        # self.feed_forward2 = nn.Sequential(nn.Linear(d_model, 4*d_model), nn.ReLU(), nn.Linear(4*d_model, d_model))
        self.linear = nn.Linear(d_model, d_model)
        self.rnn_cell = nn.GRUCell(d_model, d_model, bias=False)
        self.time_layer = nn.Linear(1, 64)
        self.time_layer2 = nn.Linear(1, 64)
        self.time_updim = nn.Linear(64, d_model)
        self.mem_linear = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.LayerNorm(d_model))
        self.step_linear = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.LayerNorm(d_model))
        self.global_att = nn.Sequential(nn.Linear(2 * d_model + 64, d_model), nn.Tanh(), nn.Linear(d_model, 1))

    def forward(self, input_seqs, lengths, seq_time_step):
        seq_time_step = seq_time_step.unsqueeze(2) / 180
        time_feature = 1 - self.tanh(torch.pow(self.time_layer(seq_time_step), 2))
        time_encoding = self.time_updim(time_feature)
        batch_size, seq_len, num_cui_per_visit = input_seqs.size()
        x = self.embedding(input_seqs)
        x = self.emb_dropout(x)
        x = self.relu(x)
        cat_features = torch.cat((self.embedding(torch.arange(0, self.vocab_size+1).to(x.device)), self.target_embedding(x.new_zeros(self.vocab_size + 1).long())), dim=-1)
        p = self.code_selection(cat_features)
        p = F.gumbel_softmax(torch.log_softmax(p, -1), hard=True)
        p_mask = p[:, 0][input_seqs]
        x_selected = p_mask.unsqueeze(-1) * x
        selected = x_selected.sum(-2) + time_encoding
        following_feature, _ = self.rnn_reverse(selected.flip([1]))
        following_feature = torch.flip(following_feature, [1])
        hx = x.new_zeros((batch_size, self.rnn_cell.hidden_size))
        hiddens = []
        v_skips = []
        c_skips = []
        for i in range(seq_len):
            v_skip = self.visit_skip(torch.cat((hx, selected[:, i], following_feature[:, i], self.target_linear2(self.target_embedding(x.new_zeros(batch_size).long()))), dim=-1))
            v_skip = F.gumbel_softmax(torch.log_softmax(v_skip, dim=-1), hard=True)
            # print(v_skip.size())
            v_skips.append(v_skip)
            code_query = self.code_query(torch.cat((hx, following_feature[:, i], self.target_linear(self.target_embedding(x.new_zeros(batch_size).long()))), dim=-1))
            scale = np.sqrt(self.d_model)
            energy = (torch.bmm(code_query.unsqueeze(1), x[:, i].permute(0, 2, 1)) / scale)
            attention = torch.softmax(energy, dim=-1) * p_mask[:, i].unsqueeze(1)
            # print(attention.size())
            c_skips.append(attention.squeeze())
            z_i = torch.matmul(attention, x[:, i]).squeeze()
            z_i = self.fc(z_i) + time_encoding[:, i]
            z_i = self.dropout2(z_i)
            # z_i = self.layer_norm(z_i)
            step_hx = self.rnn_cell(z_i, hx)
            step_hx = hx * v_skip[:, 1].unsqueeze(-1) + step_hx * v_skip[:, 0].unsqueeze(-1)
            # cx = cx * v_skip[:, 1].unsqueeze(-1) + step_cx * v_skip[:, 0].unsqueeze(-1)
            if i == 0:
                hx = step_hx
            else:
                memory = self.mem_linear(torch.stack(hiddens, dim=1))
                step_hx = self.step_linear(step_hx)
                mem_energy = torch.bmm(step_hx.unsqueeze(1), memory.permute(0, 2, 1)) / scale
                mem_attention = torch.softmax(mem_energy, dim=-1)
                hx = torch.matmul(mem_attention, memory).squeeze()
                hx = self.dropout(hx)
                hx = self.layer_norm2(hx + step_hx)
                hx = self.layer_norm(self.feed_forward(hx) + hx)
            hiddens.append(hx)
            # skips.append(v_skip[:, 1])
        hiddens = torch.stack(hiddens, dim=1)
        # v_skips = torch.stack(v_skips, dim=1)
        # c_skips = torch.stack(c_skips, dim=1)
        # one = torch.ones_like(c_skips)
        # c_skips = torch.where(c_skips > 0, one, c_skips)
        # code_mask = torch.where(code_mask > 0, one, code_mask)
        # skips = v_skips[:, :, 0].unsqueeze(-1) * c_skips
        # skip_rate = 1 - torch.divide(torch.sum(skips), torch.sum(1 - code_mask))
        length_mask = (torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len) >= lengths.unsqueeze(1))
        # hiddens = hiddens.masked_fill(length_mask.unsqueeze(-1).expand_as(hiddens), float('-inf'))
        time_feature2 = 1 - self.tanh(torch.pow(self.time_layer2(seq_time_step), 2))
        global_energy = self.global_att(torch.cat((hiddens, time_feature2, self.target_embedding(x.new_zeros(batch_size, seq_len).long())), dim=-1))
        global_energy = global_energy.masked_fill(length_mask.unsqueeze(-1).expand_as(global_energy), float('-inf'))
        global_attention = torch.softmax(global_energy, dim=1)
        out_feature = global_attention * hiddens
        # final_hidden = torch.gather(hiddens, 1, lengths.unsqueeze(1).unsqueeze(2).expand(batch_size, 1, self.rnn_cell.hidden_size) - 1).squeeze()
        out = self.output_mlp(out_feature.sum(1))
        return out