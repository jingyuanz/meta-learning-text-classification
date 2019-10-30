import learn2learn as l2l
from transformers import *
import torch as T
from torch import nn
import torch.nn.functional as F
from pytorch_pretrained_bert import BertAdam
import torchvision
from torch.utils.data import *
from learn2learn.data import MetaDataset
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.sequence import pad_sequences
device = T.device('cuda:1' if T.cuda.is_available() else 'cpu')
from learn2learn.data import TaskGenerator

def load_data():

    with open('data/train.txt','r',encoding='utf-8') as f:
        lines = f.readlines()
        lines = [x.strip().split('\t') for x in lines]
        X = [x[1] for x in lines]
        Y = [x[0] for x in lines]
        lb_encoder = LabelEncoder()
        Y = lb_encoder.fit_transform(Y)
        tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
        X = [tokenizer.encode(x, add_special_tokens=True) for x in X]
        X = pad_sequences(X,  maxlen=50, padding='post', truncating='post')
        return X, Y


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.encoder = BertModel.from_pretrained('bert-base-chinese')
        self.classifier = nn.Linear(768, 15)



    def forward(self, x):
        x = self.encoder(x)[1]
        o = self.classifier(x)
        return F.dropout(o, p=0.5)



class Trainer:
    def __init__(self):
        model = Net()
        self.loss_fn = nn.CrossEntropyLoss()
        model.to(device)
        self.meta_model = l2l.algorithms.MAML(model, lr=1e-3, first_order=True)
        self.optim = AdamW(self.meta_model.parameters(), lr=3e-5)
        # text_train = l2l.text.datasets.NewsClassification(root=download_location, download=True)
        # train_gen = l2l.text.datasets.TaskGenerator(text_train, ways=ways)
        X, Y = load_data()
        self.dataset = TensorDataset(T.LongTensor(X), T.LongTensor(Y))
        self.metaset = MetaDataset(self.dataset)
        self.task_generator = TaskGenerator(self.metaset, ways=15, shots=10, classes=None, tasks=1000)

    def accuracy(self, predictions, targets):
        predictions = predictions.argmax(dim=1)
        acc = (predictions == targets).sum().float()
        acc /= len(targets)
        return acc.item()

    def compute_loss(self, task, learner):
        dl = T.utils.data.DataLoader(task, batch_size=150, shuffle=True, num_workers=0)
        loss = 0.
        acc = 0.
        for i, (x,y) in enumerate(dl):
            x, y = x.to(device), y.long().to(device)
            out = learner(x)
            l = self.loss_fn(out, y)
            loss += l
            acc += self.accuracy(out, y)
        loss /= len(task)
        print(loss)
        acc /= len(dl)
        print(acc)
        return loss, acc

    def train(self):
        for i in range(100):
            print("EPOCH: ", i)
            meta_error = 0.
            meta_acc = 0.
            '''num clone'''
            for _ in range(3):
                learner = self.meta_model.clone()
                train_task = self.task_generator.sample(shots=10)
                test_task = self.task_generator.sample(shots=10)

                for j in range(10):
                    loss, _ = self.compute_loss(train_task, learner)
                    learner.adapt(loss)

                eval_loss, eval_acc = self.compute_loss(test_task, learner)
                meta_acc += eval_acc
                meta_error += eval_loss
            meta_error /= 3
            meta_acc /= 3
            print("Loss : {:.3f} Acc : {:.3f}".format(meta_error.item(), meta_acc))
            self.optim.zero_grad()
            meta_error.backward()
            self.optim.step()



t = Trainer()
t.train()



