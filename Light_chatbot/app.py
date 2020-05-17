from flask import Flask, request, jsonify
import torch
from torchtext import data
from generation import inference, tokenizer1
from Styling import make_special_token
from model import Transformer

app = Flask(__name__)
device = torch.device('cpu')
max_len = 40
ID = data.Field(sequential=False,
                use_vocab=False)
SA = data.Field(sequential=False,
                use_vocab=False)
TEXT = data.Field(sequential=True,
                  use_vocab=True,
                  tokenize=tokenizer1,
                  batch_first=True,
                  fix_length=max_len,
                  dtype=torch.int32
                  )

LABEL = data.Field(sequential=True,
                   use_vocab=True,
                   tokenize=tokenizer1,
                   batch_first=True,
                   fix_length=max_len,
                   init_token='<sos>',
                   eos_token='<eos>',
                   dtype=torch.int32
                   )
text_specials, label_specials = make_special_token(False)
train_data, _ = data.TabularDataset.splits(
    path='.', train='chatbot_0325_ALLLABEL_train.txt', format='tsv',
    fields=[('id', ID), ('text', TEXT), ('target_text', LABEL), ('SA', SA)], skip_header=True
)
TEXT.build_vocab(train_data, max_size=15000, specials=text_specials)
LABEL.build_vocab(train_data, max_size=15000, specials=label_specials)
soft_model = Transformer(160, 2, 2, 0.1, TEXT, LABEL)
# rough_model = Transformer(args, TEXT, LABEL)
soft_model.to(device)
# rough_model.to(device)
soft_model.load_state_dict(torch.load('sorted_model-soft.pth', map_location=device)['model_state_dict'])


# rough_model.load_state_dict(torch.load('sorted_model-rough.pth', map_location=device)['model_state_dict'])


@app.route('/soft', methods=['POST'])
def soft():
    if request.is_json():
        sentence = request.json["data"]
        return jsonify({"data": inference(device, max_len, TEXT, LABEL, soft_model, sentence)}), 200
    else:
        return jsonify({"data": "잘못된 요청입니다. Bad Request."}), 400

# @app.route('/rough', methods=['POST'])
# def rough():
#     return inference(device, max_len, TEXT, LABEL, rough_model, ), 200


if __name__ == '__main__':
    app.run()
