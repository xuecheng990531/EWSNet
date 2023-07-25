import csv

def write_to_csv(values):

    # 列名
    fieldnames = ['Epoch','Accuracy', 'Sensitivity', 'AUC','IoU','F1']

    # 写入CSV文件
    filename = 'logs/metrics.csv'

    # 检查文件是否为空
    file_empty = False
    try:
        with open(filename, 'r') as file:
            reader = csv.reader(file)
            if len(list(reader)) == 0:
                file_empty = True
    except FileNotFoundError:
        file_empty = True

    # 写入指标值
    with open(filename, 'a', newline='') as file:
        writer = csv.writer(file)
        
        # 如果文件为空，先写入列名
        if file_empty:
            writer.writerow(fieldnames)
        
        # 写入每一轮的指标值
        writer.writerow(values)
