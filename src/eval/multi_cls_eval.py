import os
import json
import sys
import sklearn.metrics as met
from sklearn.metrics import classification_report
from sklearn.metrics import hamming_loss
from sklearn.metrics import accuracy_score

import numpy as np

def multilabel_topk_accuracy(preds, targets, k):
    """
    멀티레이블 문제에서 ACC@k 계산
    :param preds: shape (N, num_classes), 각 클래스의 예측 점수 (예: 확률)
    :param targets: shape (N, num_classes), 0/1 형태의 멀티레이블 ground truth (1이면 해당 클래스가 정답)
    :param k: 상위 k개의 예측을 고려
    :return: 전체 샘플에 대한 ACC@k 값
    """
    total = 0.0
    num_samples = preds.shape[0]
    for i in range(num_samples):
    # i번째 샘플의 top-k 예측 클래스 인덱스
        topk_indices = np.argsort(preds[i])[-k:]
        # i번째 샘플의 정답 클래스 인덱스 (targets[i]가 멀티핫 벡터)
        true_indices = np.where(targets[i] == 1)[0]
        if len(true_indices) == 0:
            sample_precision = 0.0
        else:
            # 정답의 개수가 k보다 적을 경우엔 분모를 정답의 개수로 조정
            denominator = min(k, len(true_indices))
            sample_precision = len(np.intersect1d(topk_indices, true_indices)) / denominator
        # top-k 예측과 정답 간의 교집합 개수를 k로 나누어 정밀도를 구함
        total += sample_precision
    return total / num_samples



if __name__=='__main__':
    
    filename = sys.argv[1]
    ext = os.path.splitext(filename)[-1]
    if ext == '.jsonl':
        with open(filename, 'r') as f:
            data = [json.loads(d) for d in f]


        pred = [d['output']['preds'] for d in data if 'output' in d]
        gold = [d['data']['target'] for d in data if 'data' in d]
        srce = [d['output']['inputs'] for d in data if 'output' in d]
        logits = [d['output']['logit'] for d in data if 'output' in d]
        gold_list = [d['output']['labels'] for d in data if 'data' in d]

        pred = np.array(logits)
        gold = np.array(gold_list)

        acc_at_1 = multilabel_topk_accuracy(pred, gold, k=1)
        print('acc@1')
        print(acc_at_1)
        pred_to_one_hot = list()
        for i, data in enumerate(pred) :
            k = int(gold[i].sum())
            topk_index = np.argsort(pred[i])[-k:]
            binary_pred = np.zeros_like(pred[i])
            binary_pred[topk_index] = 1
            binary_pred = binary_pred.astype(int).tolist()
            pred_to_one_hot.append(binary_pred)
        gold = gold.astype(int).tolist()
    #data_dict= {27: 14219, 0: 4130, 4: 2939, 15: 2662, 3: 2470, 1: 2328, 7: 2191, 18: 2086, 10: 2022, 20: 1581, 2: 1567, 17: 1452, 6: 1368, 25: 1326, 9: 1269, 22: 1110, 5: 1087, 26: 1060, 13: 853, 11: 793, 8: 641, 14: 596, 24: 545, 12: 303, 19: 164, 23: 153, 21: 111, 16: 77}
    #sort_dict = sorted(data_dict , key=lambda x : data_dict[x])
    #head = sort_dict[:5]
    #median = sort_dict[5:23]
    #tail = sort_dict[23:]
    #print(head)
    #print(median)
    #print(tail)
    #print(sort_dict)
        
        
    #print('recall')
    #print(met.recall_score(gold,pred_to_one_hot, average='micro'))
    #report = classification_report(gold, pred_to_one_hot, digits=4, output_dict=True)
    #class_f1_scores = {int(cls): metrics["f1-score"] for cls, metrics in report.items()if cls.isdigit()}  # '0', '1', '2', ... 만 선택
    #print(class_f1_scores)
    #list_f1 = [0,0,0]

    #for i in head :
    #    print(class_f1_scores[i])
    #    list_f1[2] += class_f1_scores[i]
    #list_f1[2] = list_f1[2]/len(head)
    #for i in median :
    #    list_f1[1] += class_f1_scores[i]
    #list_f1[1] = list_f1[1]/len(median)
    #for i in tail :
    #    list_f1[0] += class_f1_scores[i]
    #list_f1[0] = list_f1[0]/len(tail)
    #print(tail)
    #print(list_f1)
    #for cls, f1 in class_f1_scores.items():
    #    print(f"Class {cls}: F1 = {f1:.4f}")
    report = classification_report(gold, pred_to_one_hot, digits=4, output_dict=False)
    hamming = hamming_loss(gold, pred_to_one_hot)
    subset_acc = accuracy_score(gold, pred_to_one_hot)
    print(report)
    print(f"Subset Accuracy: {subset_acc:.4f}")

    print(f"Hamming Loss: {hamming:.4f}")




