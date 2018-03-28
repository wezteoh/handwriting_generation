# quick and dirty data preparation script

import numpy as np
import torch

strokes = np.load('data/strokes.npy', encoding='latin1')
with open('data/sentences.txt') as f:
    texts = f.readlines()
    
train_strokes = []
train_texts = []
validation_strokes = []
validation_texts = []

# only train data with length at most 800
for _ in range(len(strokes)):
    if len(strokes[_]) <= 801:
        train_strokes.append(strokes[_])
        train_texts.append(texts[_])
    else:
        validation_strokes.append(strokes[_])
        validation_texts.append(texts[_])

# pad with zeros and build masks
train_masks = np.zeros((len(train_strokes),800))
for i in range(len(train_strokes)):
    train_masks[i][0:len(train_strokes[i])-1] = 1
    train_strokes[i] = np.vstack([train_strokes[i], np.zeros((801-len(train_strokes[i]), 3))])
    
validation_masks = np.zeros((len(validation_strokes),1200))
for i in range(len(validation_strokes)):
    validation_masks[i][0:len(validation_strokes[i])-1] = 1
    validation_strokes[i] = np.vstack([validation_strokes[i], np.zeros((1201-len(validation_strokes[i]), 3))])

np.save('data/train_strokes_800', np.stack(train_strokes))
np.save('data/train_masks_800', train_masks)
np.save('data/validation_strokes_800', np.stack(validation_strokes))
np.save('data/validation_masks_800', validation_masks)

# convert each text sentence to an array of onehots
char_list = ' ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz,."\'?-'

char_to_code = {}
code_to_char = {}
c = 0
for _ in char_list:
    char_to_code[_] = c
    code_to_char[c] = _
    c += 1
torch.save(char_to_code, 'char_to_code.pt')

max_text_len = np.max(np.array([len(a) for a in validation_texts]))

train_onehot_800 = []
train_text_masks = []
for t in train_texts:
    onehots = np.zeros((max_text_len, len(char_to_code)+1))
    mask = np.ones(max_text_len)
    for _ in range(len(t)):
        try:
            onehots[_][char_to_code[t[_]]] = 1
        except:
            onehots[_][-1] = 1
    mask[len(t):] = 0
    train_onehot_800.append(onehots)
    train_text_masks.append(mask)
train_onehot_800 = np.stack(train_onehot_800)
train_text_masks = np.stack(train_text_masks)
train_text_lens = np.array([[len(a)] for a in train_texts])

validation_onehot_800 = []
validation_text_masks = []
for t in validation_texts:
    onehots = np.zeros((max_text_len, len(char_to_code)+1))
    mask = np.ones(max_text_len)
    for _ in range(len(t)):
        try:
            onehots[_][char_to_code[t[_]]] = 1
        except:
            onehots[_][-1] = 1
    mask[len(t):] = 0
    validation_onehot_800.append(onehots)
    validation_text_masks.append(mask)
validation_onehot_800 = np.stack(validation_onehot_800)
validation_text_masks = np.stack(validation_text_masks)
validation_text_lens = np.array([[len(a)] for a in validation_texts])

np.save('data/train_onehot_800', train_onehot_800)
np.save('data/validation_onehot_800', validation_onehot_800)
np.save('data/train_text_masks', train_text_masks)
np.save('data/validation_text_masks', validation_text_masks)
np.save('data/train_text_lens', train_text_lens)
np.save('data/validation_text_lens', validation_text_lens)


