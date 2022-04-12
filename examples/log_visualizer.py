import argparse
import os

import matplotlib.pyplot as plt
import seaborn as sns


colors=['#8ECFC9','#FFBE7A','#FA7F6F','#82B0D2','#9394E7','#D76364','#54B345','#05B9E2']
def get_argparser():
    parser = argparse.ArgumentParser(description='Log visualizer')
    parser.add_argument('--logs', required=True, metavar='N', nargs='+', help='list of log file paths to visualize')
    parser.add_argument('--labels', metavar='N', nargs='+', help='list of labels used in plots')
    parser.add_argument('--task', default='classification', help='type of tasks defined in the log files')
    return parser

def Smooth_loss(loss,weight=0.85,window=2):
    scaler=loss
    last=scaler[0]
    smoothed=[]
    loss_min=[]
    loss_max=[]
    for i,point in enumerate(scaler):
        smoothed_val=last*weight+point*(1-weight)
        smoothed.append(smoothed_val)
        last=smoothed_val
        left_index=max(0,i-window)
        right_index=min(len(scaler)-1,i+window)
        selected_point=loss[left_index:right_index]
        val_min=min(min(selected_point),smoothed_val)
        val_max=max(max(selected_point),smoothed_val)
        loss_max.append(val_max)
        loss_min.append(val_min)
    return smoothed,loss_min,loss_max



def read_files(file_paths, labels):
    if labels is None or len(labels) != len(file_paths):
        labels = [os.path.basename(file_path) for file_path in file_paths]

    log_dict = dict()
    for file_path, label in zip(file_paths, labels):
        with open(os.path.expanduser(file_path), 'r') as fp:
            log_dict[file_path] = ([line.strip() for line in fp], label)
    return log_dict


def extract_train_time(message, keyword='Total time: ', sub_keyword=' day'):
    if not message.startswith('Epoch:') or keyword not in message:
        return None

    time_str = message[message.find(keyword) + len(keyword):]
    hours = 0
    if sub_keyword in time_str:
        start_idx = time_str.find(sub_keyword)
        hours = 24 * int(time_str[:start_idx])
        time_str = time_str.split(' ')[-1]
    h, m, s = map(int, time_str.split(':'))
    return ((hours + h) * 60 + m) * 60 + s


def extract_val_acc(message, acc1_str='Acc@1 '):
    if acc1_str not in message:
        return None

    acc1 = float(message[message.find(acc1_str) + len(acc1_str):])
    return acc1


def extract_val_performance(log_lines):
    train_time_list, val_acc1_list = list(), list()
    for line in log_lines:
        elements = line.split('\t')
        if len(elements) < 3:
            continue

        message = elements[3]
        train_time = extract_train_time(message)
        if isinstance(train_time, int):
            train_time_list.append(train_time)
            continue

        val_acc1 = extract_val_acc(message)
        if isinstance(val_acc1, float):
            val_acc1_list.append(val_acc1)
        if 'Training time' in message:
            break
    return train_time_list, val_acc1_list


def visualize_val_performance(log_dict):
    # sns.set()
    val_performance_dict = dict()
    # plt.figure(facecolor='white',edgecolor='black')
    for i,(file_path, (log_lines, label)) in enumerate(log_dict.items()):
        train_times, val_acc1s = extract_val_performance(log_lines)
        val_performance_dict[file_path] = (train_times, val_acc1s, label)
        xs = list(range(len(val_acc1s)))
        smoothed,loss_min,loss_max=Smooth_loss(val_acc1s,window=5)
        print(len(loss_min),len(loss_max),len(smoothed))
        plt.fill_between(xs,loss_min,loss_max,where=[i<j for i,j in zip(loss_min,loss_max)],alpha=0.1,interpolate=True,facecolor=colors[i])
        plt.plot(xs, smoothed, label=r'${}$'.format(label),color=colors[i],linewidth=2)
    ax=plt.gca()
    ax.patch.set_facecolor("white")
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_facecolor('white')
    # plt.grid(True)
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Top-1 Validation Accuracy [%]')
    plt.xlim(80,188)
    plt.ylim(82,93)
    plt.tight_layout()
    plt.show()

    # for file_path, (train_times, val_acc1s, label) in val_performance_dict.items():
    #     accum_train_times = [sum(train_times[:i + 1]) for i in range(len(train_times))]
    #     plt.plot(accum_train_times, val_acc1s, '-o', label=r'${}$'.format(label))
    #
    # plt.legend()
    # plt.xlabel('Training time [sec]')
    # plt.ylabel('Top-1 Validation Accuracy [%]')
    # plt.tight_layout()
    # plt.show()


def main(args):
    log_dict = read_files(args.logs, args.labels)
    task = args.task
    if task == 'classification':
        visualize_val_performance(log_dict)


if __name__ == '__main__':
    argparser = get_argparser()
    main(argparser.parse_args())

"""
 python log_visualizer.py --logs /home/qiuziming/product/torchdistill/log/cifar10/kd/fkd/resnet18_from_resnet50_origin_fc_fm.txt /home/qiuziming/pr
oduct/torchdistill/log/cifar10/kd/fkd/resnet18_from_resnet50_origin_layer1_c.txt /home/qiuziming/product/torchdistill/log/cifar10/kd/fkd/resnet18_from_resnet50_origin_layer1_fm.txt /home/qiuziming/product/torchdistill/log/cifar10/kd/fkd/resnet18_from_resnet50_origin_layer2_c.txt /home/qiuziming/product/torchdistill/log/cifar10/kd/fkd/resnet18_from_resnet50_origin_layer2_fm.txt /home/qiuziming/product/torchdistill/log/cifar10/kd/fkd/resnet18_from_resnet50_origin_layer3_c.txt /home/qiuziming/product/torchdistill/log/cifar10/kd/fkd/resnet18_from_resnet50_origin_layer3_fm.txt --labels FC Layer1-Channel Layer1-FeatureMap  Layer2-Channel Layer2-FeatureMap  Layer3-Channel Layer3-FeatureMap

"""