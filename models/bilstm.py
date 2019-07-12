import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from torch.nn import  functional as F

class BiLSTM(nn.Module):
    def __init__(self, vocab_size, emb_size,weight, hidden_size, out_size):
        """初始化参数：
            vocab_size:字典的大小
            emb_size:词向量的维数
            hidden_size：隐向量的维数
            out_size:标注的种类
        """
        super(BiLSTM, self).__init__()
        self.preTrainembedding = nn.Embedding(vocab_size, 300)
        self.preTrainembedding.weight.data.copy_(torch.Tensor(weight))
        self.embed = nn.Embedding(vocab_size, emb_size)
        # self.embedding.weight.data.copy_(torch.Tensor(weight))
        # self.dropout1=nn.Dropout(0.5)
        self.lstm = nn.LSTM(emb_size, hidden_size,
                              batch_first=True,
                              bidirectional=True,num_layers=3,dropout=0.5)
        self.embed_drop = nn.Dropout(0.8)
        self.conv1=nn.Conv1d(emb_size,100,3,padding=1)
        self.dense1=nn.Linear(50,50)
        self.conv2 = nn.Conv1d(emb_size, 100, 5, padding=2)
        self.dense2 = nn.Linear(50, 50)
        # self.conv3 = nn.Conv1d(emb_size, 50, 7, padding=3)
        # self.dense4 = nn.Linear(50, 50)
        self.conv4 = nn.Conv1d(emb_size, 100, 1, padding=0)
        self.dense3 = nn.Linear(50, 50)
        self.lin = nn.Linear(2 * hidden_size, hidden_size)
        self.lin_drop=nn.Dropout(0.5)
        self.output = nn.Linear(hidden_size, out_size)

        # self.dropout2=nn.Dropout(0.2)
    def forward(self, sents_tensor, lengths):
        # print(sents_tensor.size(),lengths)
        emb = self.preTrainembedding(sents_tensor)  # [B, L, emb_size]
        emb = self.embed_drop(emb)

        # conv1=self.conv1(emb.permute([0, 2, 1])).permute([0, 2, 1])

        # conv1=self.dense1(conv1)
        # conv2 =self.conv2(emb.permute([0, 2, 1])).permute([0, 2, 1])

        # conv2 = self.dense2(conv2)
        # conv3 =self.conv3(emb.permute([0, 2, 1])).permute([0, 2, 1])
        #
        # conv3 = self.dense3(conv3)
        # conv4 =self.conv4(emb.permute([0, 2, 1])).permute([0, 2, 1])
        # conv4 = self.dense4(conv4)

        # conv=torch.cat([conv1,conv2,conv4], dim=2)
        packed = pack_padded_sequence(emb, lengths, batch_first=True)
        rnn_out, _ = self.lstm(packed)
        # rnn_out=torch.cat([rnn_out,conv],dim=2)
        # rnn_out:[B, L, hidden_size*2]
        rnn_out, _ = pad_packed_sequence(rnn_out, batch_first=True)
        lin = self.lin(rnn_out) # [B, L, out_size]
        lin=self.lin_drop(lin)
        scores=self.output(lin)
        return scores

    def test(self, sents_tensor, lengths, _):
        """第三个参数不会用到，加它是为了与BiLSTM_CRF保持同样的接口"""
        logits = self.forward(sents_tensor, lengths)  # [B, L, out_size]
        _, batch_tagids = torch.max(logits, dim=2)

        return batch_tagids
