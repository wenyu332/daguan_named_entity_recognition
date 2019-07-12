import time
from collections import Counter

from models.bilstm_crf import BILSTM_Model
from utils import save_model, flatten_lists,get_ner_fmeasure
import numpy as np
import torch



def bilstm_train_and_eval(train_data, dev_data, test_data,
                          charEmbedding,word2id, tag2id, crf=True, remove_O=False):
    train_word_lists, train_tag_lists = train_data
    dev_word_lists, dev_tag_lists = dev_data
    test_word_lists, test_tag_lists = test_data
    start = time.time()
    vocab_size = len(word2id)
    out_size = len(tag2id)
    bilstm_model = BILSTM_Model(vocab_size,charEmbedding, out_size, crf=crf)
    bilstm_model.train(train_word_lists, train_tag_lists,
                       dev_word_lists, dev_tag_lists, word2id, tag2id)

    model_name = "bilstm_crf" if crf else "bilstm"
    save_model(bilstm_model, "./ckpts/"+str(bilstm_model.best_val_loss)[:5]+model_name+".pkl")

    print("训练完毕,共用时{}秒.".format(int(time.time()-start)))
    print("评估{}模型中...".format(model_name))
    pred_tag_lists, test_tag_lists = bilstm_model.test(
        test_word_lists, test_tag_lists, word2id, tag2id)
    f = open('next_dev_result/'+str(bilstm_model.best_val_loss)[:5]+'_bilstmcrf_result.txt', 'w')
    for pred_tag_list in pred_tag_lists:
        f.write(' '.join(pred_tag_list) + '\n')
    f.close()
    # metrics=get_ner_fmeasure(test_tag_lists,pred_tag_lists)
    # f=open('previous_dev_results.txt','a')
    # f.write(' '.join(metrics)+'\n')
    # f.close()
    # metrics = Metrics(test_tag_lists, pred_tag_lists, remove_O=remove_O)
    # metrics.report_scores()
    # metrics.report_confusion_matrix()

    # return metrics


def ensemble_evaluate(results, targets, remove_O=False):
    """ensemble多个模型"""
    for i in range(len(results)):
        results[i] = flatten_lists(results[i])

    pred_tags = []
    for result in zip(*results):
        ensemble_tag = Counter(result).most_common(1)[0][0]
        pred_tags.append(ensemble_tag)

    targets = flatten_lists(targets)
    assert len(pred_tags) == len(targets)

    print("Ensemble 四个模型的结果如下：")
    metrics = Metrics(targets, pred_tags, remove_O=remove_O)
    metrics.report_scores()
    metrics.report_confusion_matrix()
