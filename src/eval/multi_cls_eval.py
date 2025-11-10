import os
import json
import sys
import sklearn.metrics as met
from sklearn.metrics import classification_report
from sklearn.metrics import hamming_loss

import numpy as np




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

        pred_to_one_hot = list()
        for i, data in enumerate(pred) :
            k = int(gold[i].sum())
            topk_index = np.argsort(pred[i])[-k:]
            binary_pred = np.zeros_like(pred[i])
            binary_pred[topk_index] = 1
            binary_pred = binary_pred.astype(int).tolist()
            pred_to_one_hot.append(binary_pred)
        gold = gold.astype(int).tolist()
    report = classification_report(gold, pred_to_one_hot, digits=4, output_dict=False)
    hamming = hamming_loss(gold, pred_to_one_hot)
    print(report)
    print(f"Hamming Loss: {hamming:.4f}")




