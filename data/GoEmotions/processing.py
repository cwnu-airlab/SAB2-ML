import sys
import os
import json



def processing(path, save):
    temp_list = list()

    with open(path,'r') as f:
        while True:

            data = f.readline()
            if not data:
                break
            print(data)
            data_list = data.split('\t')
            print(data_list)
            source = data_list[0]
            target = [int(x) for x in data_list[1].strip('\n').split(',')]
            data_dict = {'source' : source,  'target' : target}
            temp_list.append(data_dict)

    with open(save, 'w') as f :
        for item in temp_list :
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


if __name__ == "__main__":
    path = sys.argv[1]
    save = sys.argv[2]
    
    processing(path, save)

