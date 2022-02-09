import argparse
from os import name
import matplotlib.pyplot as plt

def parse_args():
    
    parser = argparse.ArgumentParser(description='LOG file parser')

    parser.add_argument('--log_file', help='path to file with logging information',
                        required=True, type=str)

    parser.add_argument('--result', help='path to save the loss graph',
                        required=True, type=str)
                                              
    args = parser.parse_args()

    return args

def parse_log_file(log_file):

    train_loss, val_loss = [], []
    train_nme, val_nme = [], []

    with open(log_file) as file:

        file_lines = file.readlines()

        for line in file_lines:

            if 'Train Epoch' in line:

                tmp = line.split('loss:')[1]
                tmp = tmp.split('nme:')
        
                train_loss.append(float(tmp[0]))
                train_nme.append(float(tmp[1]))
            
            elif 'Test Epoch' in line:
                
                tmp = line.split('loss:')[1]
                tmp = tmp.split('nme:')

                val_loss.append(float(tmp[0]))
                tmp = tmp[1].split('[')
                val_nme.append(float(tmp[0]))

    return train_loss, val_loss, train_nme, val_nme


def build_graph(path_to_graph, train_loss, val_loss, train_nme, val_nme):
    
    epochs = list(range(len(train_loss)))

    plt.plot(epochs, train_loss, label = 'train loss')
    plt.plot(epochs, val_loss, label = 'val loss')

    plt.ylabel('loss')
    plt.xlabel('epoch')

    plt.title('loss')
    plt.legend()

    plt.savefig(path_to_graph + 'loss.png')
    
    plt.close()

    plt.plot(epochs, train_nme, label = 'train nme')
    plt.plot(epochs, val_nme, label = 'val nme')


    plt.ylabel('nme')
    plt.xlabel('epoch')

    plt.title('nme')
    plt.legend()

    plt.savefig(path_to_graph + 'nme.png')



def main ():

    args = parse_args()

    train_loss, val_loss, train_nme, val_nme = parse_log_file(args.log_file)

    build_graph(args.result, train_loss, val_loss, train_nme, val_nme)


if __name__ == '__main__':
    main()
