import torch

# print(torch.cuda.is_available())


label_len = 36
vocab =  "<,.+:-?$ 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ>"
# start symbol <
# end symbol >
char2token = {"PAD":0}
token2char = {0:"PAD"}
for i, c in enumerate(vocab):
    char2token[c] = i+1
    token2char[i+1] = c

def illegal(label):
    if len(label) > label_len-1:
        return True
    for l in label:
        if l not in vocab[1:-1]:
            return True
    return False

num_lines=[]
with open('ListDataset.txt') as f:
    for lines in f:
        flines = f.readlines()
        # for i in flines:
        #     if not illegal(i.strip('\n').split('\\t')[1]):
        #         print(i.strip('\n').split('\\t')[1])
        num_lines += [i for i in flines if not illegal(i.strip('\n').split('\\t')[1])]
        print(len(num_lines))
        # print('\n'.join(map(str, num_lines)))