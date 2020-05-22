import torch
from konlpy.tag import Mecab
from torch.autograd import Variable
from chatspace import ChatSpace

spacer = ChatSpace()


def tokenizer1(text: str):
    result_text = ''.join(c for c in text if c.isalnum())
    a = Mecab().morphs(result_text)
    return [a[i] for i in range(len(a))]


def inference(device: torch.device, max_len: int, TEXT, LABEL, model: torch.nn.Module, sentence: str):

    enc_input = tokenizer1(sentence)
    enc_input_index = []

    for tok in enc_input:
        enc_input_index.append(TEXT.vocab.stoi[tok])

    for j in range(max_len - len(enc_input_index)):
        enc_input_index.append(TEXT.vocab.stoi['<pad>'])

    enc_input_index = Variable(torch.LongTensor([enc_input_index]))

    dec_input = torch.LongTensor([[LABEL.vocab.stoi['<sos>']]])

    model.eval()
    pred = []
    for i in range(max_len):
        y_pred = model(enc_input_index.to(device), dec_input.to(device))
        y_pred_ids = y_pred.max(dim=-1)[1]
        if y_pred_ids[0, -1] == LABEL.vocab.stoi['<eos>']:
            y_pred_ids = y_pred_ids.squeeze(0)
            print(">", end=" ")
            for idx in range(len(y_pred_ids)):
                if LABEL.vocab.itos[y_pred_ids[idx]] == '<eos>':
                    pred_sentence = "".join(pred)
                    pred_str = spacer.space(pred_sentence)
                    return pred_str
                else:
                    pred.append(LABEL.vocab.itos[y_pred_ids[idx]])
            return 'Error: Sentence is not end'

        dec_input = torch.cat(
            [dec_input.to(torch.device('cpu')),
             y_pred_ids[0, -1].unsqueeze(0).unsqueeze(0).to(torch.device('cpu'))], dim=-1)
    return 'Error: Sentence is not predicted'
