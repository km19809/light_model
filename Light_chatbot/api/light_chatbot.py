import argparse
import time
import torch
from torch import nn
from torchtext import data
from torchtext.data import BucketIterator
from torchtext.data import TabularDataset

from Styling import styling, make_special_token
from generation import inference, tokenizer1
from model import Transformer, GradualWarmupScheduler

SEED = 1234




def acc(yhat: torch.Tensor, y: torch.Tensor):
    with torch.no_grad():
        yhat = yhat.max(dim=-1)[1]  # [0]: max value, [1]: index of max value
        _acc = (yhat == y).float()[y != 1].mean()  # padding은 acc에서 제거
    return _acc


def train(model: Transformer, iterator, optimizer, criterion: nn.CrossEntropyLoss, max_len: int, per_soft: bool, per_rough: bool):
    total_loss = 0
    iter_num = 0
    tr_acc = 0
    model.train()

    for step, batch in enumerate(iterator):
        optimizer.zero_grad()

        enc_input, dec_input, enc_label = batch.text, batch.target_text, batch.SA
        dec_output = dec_input[:, 1:]
        dec_outputs = torch.zeros(dec_output.size(0), max_len).type_as(dec_input.data)

        # emotion 과 체를 반영
        enc_input, dec_input, dec_outputs = \
            styling(enc_input, dec_input, dec_output, dec_outputs, enc_label, max_len, per_soft, per_rough, TEXT, LABEL)

        y_pred = model(enc_input, dec_input)

        y_pred = y_pred.reshape(-1, y_pred.size(-1))
        dec_output = dec_outputs.view(-1).long()

        # padding 제외한 value index 추출
        real_value_index = [dec_output != 1]  # <pad> == 1

        # padding 은 loss 계산시 제외
        loss = criterion(y_pred[real_value_index], dec_output[real_value_index])
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            train_acc = acc(y_pred, dec_output)

        total_loss += loss
        iter_num += 1
        tr_acc += train_acc

    return total_loss.data.cpu().numpy() / iter_num, tr_acc.data.cpu().numpy() / iter_num


def test(model: Transformer, iterator, criterion: nn.CrossEntropyLoss):
    total_loss = 0
    iter_num = 0
    te_acc = 0
    model.eval()

    with torch.no_grad():
        for batch in iterator:
            enc_input, dec_input, enc_label = batch.text, batch.target_text, batch.SA
            dec_output = dec_input[:, 1:]
            dec_outputs = torch.zeros(dec_output.size(0), args.max_len).type_as(dec_input.data)

            # emotion 과 체를 반영
            enc_input, dec_input, dec_outputs = \
                styling(enc_input, dec_input, dec_output, dec_outputs, enc_label, args.max_len, args.per_soft, args.per_rough, TEXT, LABEL)

            y_pred = model(enc_input, dec_input)

            y_pred = y_pred.reshape(-1, y_pred.size(-1))
            dec_output = dec_outputs.view(-1).long()

            real_value_index = [dec_output != 1]  # <pad> == 1

            loss = criterion(y_pred[real_value_index], dec_output[real_value_index])

            with torch.no_grad():
                test_acc = acc(y_pred, dec_output)
            total_loss += loss
            iter_num += 1
            te_acc += test_acc

    return total_loss.data.cpu().numpy() / iter_num, te_acc.data.cpu().numpy() / iter_num


# 데이터 전처리 및 loader return
def data_preprocessing(args, device):
    # ID는 사용하지 않음. SA는 Sentiment Analysis 라벨(0,1) 임.
    ID = data.Field(sequential=False,
                    use_vocab=False)

    TEXT = data.Field(sequential=True,
                      use_vocab=True,
                      tokenize=tokenizer1,
                      batch_first=True,
                      fix_length=args.max_len,
                      dtype=torch.int32
                      )

    LABEL = data.Field(sequential=True,
                       use_vocab=True,
                       tokenize=tokenizer1,
                       batch_first=True,
                       fix_length=args.max_len,
                       init_token='<sos>',
                       eos_token='<eos>',
                       dtype=torch.int32
                       )

    SA = data.Field(sequential=False,
                    use_vocab=False)

    train_data, test_data = TabularDataset.splits(
        path='.', train='chatbot_0325_ALLLABEL_train.txt', test='chatbot_0325_ALLLABEL_test.txt', format='tsv',
        fields=[('id', ID), ('text', TEXT), ('target_text', LABEL), ('SA', SA)], skip_header=True
    )

    # TEXT, LABEL 에 필요한 special token 만듦.
    text_specials, label_specials = make_special_token(args.per_rough)

    TEXT.build_vocab(train_data, max_size=15000, specials=text_specials)
    LABEL.build_vocab(train_data, max_size=15000, specials=label_specials)

    train_loader = BucketIterator(dataset=train_data, batch_size=args.batch_size, device=device, shuffle=True)
    test_loader = BucketIterator(dataset=test_data, batch_size=args.batch_size, device=device, shuffle=True)

    return TEXT, LABEL, train_loader, test_loader


def main(TEXT, LABEL, arguments):

    # print argparse
    for idx, (key, value) in enumerate(args.__dict__.items()):
        if idx == 0:
            print("\nargparse{\n", "\t", key, ":", value)
        elif idx == len(args.__dict__) - 1:
            print("\t", key, ":", value, "\n}")
        else:
            print("\t", key, ":", value)

    model = Transformer(args.embedding_dim, args.nhead, args.nlayers, args.dropout, TEXT, LABEL)
    criterion = nn.CrossEntropyLoss(ignore_index=LABEL.vocab.stoi['<pad>'])
    optimizer = torch.optim.Adam(params=model.parameters(), lr=arguments.lr)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=8, total_epoch=arguments.num_epochs)
    if args.per_soft:
        sorted_path = 'sorted_model-soft.pth'
    else:
        sorted_path = 'sorted_model-rough.pth'
    model.to(device)
    if arguments.train:
        best_valid_loss = float('inf')
        for epoch in range(arguments.num_epochs):
            torch.manual_seed(SEED)
            start_time = time.time()

            # train, validation
            train_loss, train_acc = \
                train(model, train_loader, optimizer, criterion, arguments.max_len, arguments.per_soft,
                      arguments.per_rough)
            valid_loss, valid_acc = test(model, test_loader, criterion)

            scheduler.step(epoch)
            # time cal
            end_time = time.time()
            elapsed_time = end_time - start_time
            epoch_mins = int(elapsed_time / 60)
            epoch_secs = int(elapsed_time - (epoch_mins * 60))

            # torch.save(model.state_dict(), sorted_path) # for some overfitting
            # 전에 학습된 loss 보다 현재 loss 가 더 낮을시 모델 저장.
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': valid_loss},
                    sorted_path)
                print(f'\t## SAVE valid_loss: {valid_loss:.3f} | valid_acc: {valid_acc:.3f} ##')

            # print loss and acc
            print(f'\n\t==Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s==')
            print(f'\t==Train Loss: {train_loss:.3f} | Train_acc: {train_acc:.3f}==')
            print(f'\t==Valid Loss: {valid_loss:.3f} | Valid_acc: {valid_acc:.3f}==\n')



    checkpoint = torch.load(sorted_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_loss, test_acc = test(model, test_loader, criterion)  # 아
    print(f'==test_loss : {test_loss:.3f} | test_acc: {test_acc:.3f}==')
    print("\t-----------------------------")
    while True:
        sentence = input("문장을 입력하세요 : ")
        print(inference(device, args.max_len, TEXT, LABEL, model, sentence))
        print("\n")


if __name__ == '__main__':
    # argparse 정의
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_len', type=int, default=40)  # max_len 크게 해야 오류 안 생김.
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_epochs', type=int, default=22)
    parser.add_argument('--warming_up_epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--embedding_dim', type=int, default=160)
    parser.add_argument('--nlayers', type=int, default=2)
    parser.add_argument('--nhead', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--train', action="store_true")
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--per_soft', action="store_true")
    group.add_argument('--per_rough', action="store_true")
    args = parser.parse_args()
    print("-준비중-")
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    TEXT, LABEL, train_loader, test_loader = data_preprocessing(args, device)
    main(TEXT, LABEL, args)
