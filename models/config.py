# 设置lstm训练参数
class TrainingConfig(object):
    batch_size = 50
    # 学习速率
    lr = 1.0
    epoches =50
    print_step = 5


class LSTMConfig(object):
    emb_size =300  # 词向量的维数
    hidden_size = 128 # lstm隐向量的维数

    #84.91   83.27
#emb_size=80 hidden=100  conv=50  f1=82.8576
#emb_size=64 hidden=67.76 GiB total capacity;4  conv=64  f1=80.828
#emb_size=80 hidden=100 conv=64  f1=82.28  f1=84.72      f1=82.10 f1=84.59   3 3 5 f1=82.76 f1=84.72
#3 5 5 不使用激活函数f1=83.39 f1=85.18  測試了卷積之後加Relu激活函數,實驗結果爲f1=81.87  使用tanh函数 f1=83.78  f1=84.32
#3 5 5 随机80维向量与预训练的50维拼接并使用dropout=0.5 卷积层使用tanh激活函数 f1=84.92
#batch=50
# 3 5 5 随机80维向量使用dropout再与预训练的50维拼接 卷积层使用tanh激活函数 /前3000 dev=84.62 test=83.81 /后3000 test=85.47 dev=82.21
# 3 5 5 随机80维向量使用dropout再与预训练的50维拼接 卷积层不使用激活函数 /前3000 dev=84.33  test=82.75  /后3000 dev=82.37  test=84.21
# 3 5 5 随机80维向量使用dropout再与预训练的50维拼接 卷积层不使用激活函数,线性层使用Tanh  /前3000 dev=84.72  test=82.98  /后3000 dev=83.36  test=83.86
#batch=50
# 3 5 5 随机50维向量使用dropout再与预训练的50维拼接 卷积层64维,使用tanh激活函数 test=82.69   test=82.79  dev=85.97  dev=81.76  test=83.89
#3 5 5 /前3000 test=82.86  dev=84.62