"""
The synthetic_classification dataset is generated following the setup in 'Federated Optimization in Heterogeneous Networks'.
(The link for this paper is https://arxiv.org/abs/1812.06127).
The details of this dataset is described as below:

    for each user u_k:
        The data satisfies 'y_ki = argmax(softmax(W_k x_ki + b_k))'
        'x_ki:60×1, y_ki:1×1, W_k:10×60, b_k:10×1' 
        where for the locally optimal model (i.e. P(Y_k|X_k)): 
            W_k~N(u_k, 1), b_k~N(u_k, 1), u_k~N(0, alpha)
        and for the local data D_k={(x_ki, y_ki)} (i.e. P(X_k)): 
            X_k~N(v_k, Sigma), v_k[j]~N(B_k, 1) for all j in range(Dimension), B_k~N(0, beta), Sigma=Diag([i^(-1.2) for i in range(Dimension)])

    Thus, alpha controls how differently the features are distributed (i.e. P(X)), 
    and beta controls how differently the locally optimal models are distributed (i.e. P(Y|X)).

    By the way, alpha=0 and beta=0 don't means an I.I.D. distribution,
    since the local distribution of the data and the models can still
    be different (e.g. the different centers for their local Gaussian distributions).
    The I.I.D. samples are generated by using the same global model
    (i.e. keeps P(Y|X) and P(X) always the same for all the clients 
    by setting (W_k,b_k)=(W_global, b_global) and v_k=v_global = [0,...,0] for all k).

    note: The experimental setup in 'Federated Optimization in Heterogeneous Networks' adaptively set
    (alpha, beta)=(0, 0), (0.5, 0.5), (1, 1), each of which is a mixture of feature-skew and
    concept-skew according to 'Advances and Open Problems in Federated Learning' at https://arxiv.org/abs/1912.04977.
    Here, for the consistency of ('dist_id','skewness') with (alpha, beta), we consider their relationship as:
       ------------------------------------------------------------------
        ('dist_id','skewness')                  (alpha, beta)
       ------------------------------------------------------------------
                ( 0, 0)              IID: model, data global, balance
                ( 5, x)              alpha=x, model global, balance
                ( 6, x)              data global, model glabal, imbalance
                ( 8, x)              data global, beta=x, balance
                ( 9, x)              alpha=x, beta=x, balance
                (10, x)              alpha=x, beta=x, imbalance
      -------------------------------------------------------------------
    For example, the corresponding setup in our implemention is
    (dist_id, skewness) in {(6, 0), (10, 0), (10, 0.5), (10, 1.0)} as the orininal setup
    IID or (alpha, beta) in {(0,0), (0.5, 0.5), (1, 1)}.
"""
from benchmark.toolkits import BasicTaskGen
from benchmark.toolkits import XYTaskPipe
from benchmark.toolkits import ClassificationCalculator as TaskCalculator
import numpy as np
import os.path
import ujson
import random

class TaskGen(BasicTaskGen):
    def __init__(self, num_classes=10, dimension=60, dist_id = 0, num_clients = 30, skewness = 0.5, minvol=800, rawdata_path ='./benchmark/RAW_DATA/SYNTHETIC', seed=0, option=None):
        super(TaskGen, self).__init__(benchmark='synthetic_classification',
                                      dist_id=dist_id,
                                      skewness=skewness,
                                      rawdata_path=rawdata_path,
                                      seed=seed
                                      )
        self.dimension = dimension
        self.num_classes = num_classes
        self.num_clients = num_clients
        self.minvol = minvol
        self.taskname = self.get_taskname()
        self.save_task = TaskPipe.save_task
        self.taskpath = os.path.join(self.task_rootpath, self.taskname)

    def run(self):
        if not self._check_task_exist():
            self.create_task_directories()
        else:
            print("Task Already Exists.")
            return

        xs, ys, optimals = self.gen_data(self.num_clients)
        x_trains = [di[:int(0.70 * len(di))] for di in xs]
        y_trains = [di[:int(0.70 * len(di))] for di in ys]
        x_valids = [di[int(0.70 * len(di)):int(0.85 * len(di))] for di in xs]
        y_valids = [di[int(0.70 * len(di)):int(0.85 * len(di))] for di in ys]
        x_tests = [di[int(0.85 * len(di)):] for di in xs]
        y_tests = [di[int(0.85 * len(di)):] for di in ys]
        self.cnames = self.get_client_names()
        X_test = []
        Y_test = []
        for i in range(len(y_tests)):
            X_test.extend(x_tests[i])
            Y_test.extend(y_tests[i])
        self.test_data = {'x': X_test, 'y': Y_test}
        feddata = {
            'store': 'XY',
            'client_names': self.cnames,
            'dtest': self.test_data
        }
        for cid in range(self.num_clients):
            feddata[self.cnames[cid]] = {
                'dtrain': {
                    'x': x_trains[cid],
                    'y': y_trains[cid]
                },
                'dvalid': {
                    'x': x_valids[cid],
                    'y': y_valids[cid]
                },
                'optimal': optimals[cid]
            }
        with open(os.path.join(self.taskpath, 'data.json'), 'w') as outf:
            ujson.dump(feddata, outf)
        # x_trains = list()
        # y_trains = list()
        # x_tests = list()
        # y_tests = list()
        # for di in xs:
        #     x_trains.extend(di[:200])
        #     x_tests.extend(di[200:])
        # for di in ys:
        #     y_trains.extend(di[:200])
        #     y_tests.extend(di[200:])
        # x_trains = np.array(x_trains)
        # y_trains = np.array(y_trains)
        # x_tests = np.array(x_tests)
        # y_tests = np.array(y_tests)
        # print(x_trains.shape, y_trains.shape, x_tests.shape, y_tests.shape)
        # print(x_trains.dtype, y_trains.dtype, x_tests.dtype, y_tests.dtype)
        # with open(os.path.join(self.taskpath, 'x_trains.csv'), 'w') as outf:
        #     np.savetxt(outf, x_trains, delimiter=',')
        # with open(os.path.join(self.taskpath, 'y_trains.csv'), 'w') as outf:
        #     np.savetxt(outf, y_trains, delimiter=',')
        # with open(os.path.join(self.taskpath, 'x_tests.csv'), 'w') as outf:
        #     np.savetxt(outf, x_tests, delimiter=',')
        # with open(os.path.join(self.taskpath, 'y_tests.csv'), 'w') as outf:
        #     np.savetxt(outf, y_tests, delimiter=',')
        # with open(os.path.join(self.taskpath, 'partition_noniid_synthetic_nclient300.npy'), 'wb') as f:
        #     for i in range(self.num_clients):
        #         np.save(f, np.arange(200 * i, 200 * (i + 1)))
        
        

    def softmax(self, x):
        ex = np.exp(x)
        sum_ex = np.sum(np.exp(x))
        return ex / sum_ex

    def gen_data(self, num_clients):
        self.dimension = 60
        # global variables
        W_global = np.random.normal(0, 1, (self.dimension, self.num_classes))
        b_global = np.random.normal(0, 1, self.num_classes)
        v_global = np.zeros(self.dimension)
        # create Sigma = Diag([i^(-1.2) for i in range(60)])
        diagonal = np.zeros(self.dimension)
        for j in range(self.dimension):
            diagonal[j] = np.power((j + 1), -1.2)
        Sigma = np.diag(diagonal)
        # V
        V = np.zeros((num_clients, self.dimension))

        if self.dist_id == 0:
            """I.I.D. && Balance"""
            W = [W_global for _ in range(num_clients)]
            b = [b_global for _ in range(num_clients)]
            for k in range(num_clients): V[k]=v_global
            samples_per_user = [40 * self.minvol for _ in range(self.num_clients)]

        elif self.dist_id == 5:
            """Feature skew && Balance"""
            W = [W_global for _ in range(num_clients)]
            b = [b_global for _ in range(num_clients)]
            B = np.random.normal(0, self.skewness, num_clients)
            for k in range(num_clients): V[k] = np.random.normal(B[k], 1, self.dimension)
            samples_per_user = [40 * self.minvol for _ in range(self.num_clients)]

        elif self.dist_id == 6:
            """I.I.D. && Imbalance"""
            W = [W_global for _ in range(num_clients)]
            b = [b_global for _ in range(num_clients)]
            for k in range(num_clients): V[k] = v_global
            samples_per_user = np.random.lognormal(4, 2, (num_clients)).astype(int) + self.minvol

        elif self.dist_id == 8:
            """Concept skew && Balance"""
            Us = np.random.normal(0, self.skewness, num_clients)
            W = [np.random.normal(Us[k], 1, (self.dimension, self.num_classes)) for k in range(num_clients)]
            b = [np.random.normal(Us[k], 1, self.num_classes) for k in range(num_clients)]
            for k in range(num_clients): V[k] = v_global
            samples_per_user = [40 * self.minvol for _ in range(self.num_clients)]
        elif self.dist_id == 9:
            """Concept skew && Feature skew && Balance"""
            Us = np.random.normal(0, self.skewness, num_clients)
            W = [np.random.normal(Us[k], 1, (self.dimension, self.num_classes)) for k in range(num_clients)]
            b = [np.random.normal(Us[k], 1, self.num_classes) for k in range(num_clients)]
            B = np.random.normal(0, self.skewness, num_clients)
            for k in range(num_clients): V[k] = np.random.normal(B[k], 1, self.dimension)
            samples_per_user = [40 * self.minvol for _ in range(self.num_clients)]
        elif self.dist_id == 10:
            """Concept skew && Feature skew && Imbalance"""
            Us = np.random.normal(0, self.skewness, num_clients)
            W = [np.random.normal(Us[k], 1, (self.dimension, self.num_classes)) for k in range(num_clients)]
            b = [np.random.normal(Us[k], 1, self.num_classes) for k in range(num_clients)]
            B = np.random.normal(0, self.skewness, num_clients)
            for k in range(num_clients): V[k] = np.random.normal(B[k], 1, self.dimension)
            samples_per_user = np.random.lognormal(4, 2, (num_clients)).astype(int) + self.minvol
        elif self.dist_id == 11:
            """I.I.D. && Imbalance by Zipf distribution"""
            W = [W_global for _ in range(num_clients)]
            b = [b_global for _ in range(num_clients)]
            for k in range(num_clients): V[k] = v_global
            p = np.arange(self.num_clients) + 1
            p = p ** (- self.skewness)
            p = p / p.sum()
            samples_per_user = (p * 10000).astype(np.int64)
        elif self.dist_id == 12:
            """Custom"""
            Us = np.random.normal(0, self.skewness, num_clients)
            W = [np.random.normal(Us[k], 1, (self.dimension, self.num_classes)) for k in range(num_clients)]
            b = [np.random.normal(Us[k], 1, self.num_classes) for k in range(num_clients)]
            B = np.random.normal(0, self.skewness, num_clients)
            for k in range(num_clients): V[k] = np.random.normal(B[k], 1, self.dimension)
            samples_per_user = np.array([233, 233, 234] * 100)

        X_split = [[] for _ in range(num_clients)]
        y_split = [[] for _ in range(num_clients)]
        optimal_local = [np.concatenate((wk,bk.reshape(1, bk.shape[0])), axis=0).tolist() for wk, bk in zip(W, b)]

        for k in range(num_clients):
            # X_ki~N(v_k, Sigma)
            X_k = np.random.multivariate_normal(V[k], Sigma, samples_per_user[k])
            Y_k = np.zeros(samples_per_user[k], dtype=int)
            for i in range(samples_per_user[k]):
                # Y_ki = argmax(softmax(W_k x_ki + b_k))
                tmp = np.dot(X_k[i], W[k]) + b[k]
                Y_k[i] = np.argmax(self.softmax(tmp))
            X_split[k] = X_k.tolist()
            y_split[k] = Y_k.tolist()
        return X_split, y_split, optimal_local


class TaskPipe(XYTaskPipe):
    @classmethod
    def load_task(cls, task_path, data_path=None):
        with open(os.path.join(task_path, 'data.json'), 'r') as inf:
            feddata = ujson.load(inf)
        test_data = cls.TaskDataset(feddata['dtest']['x'], feddata['dtest']['y'])
        train_datas = []
        valid_datas = []
        for name in feddata['client_names']:
            train_x, train_y = feddata[name]['dtrain']['x'], feddata[name]['dtrain']['y']
            valid_x, valid_y = feddata[name]['dvalid']['x'], feddata[name]['dvalid']['y']
            if cls._cross_validation:
                k = len(train_y)
                train_x.extend(valid_x)
                train_y.extend(valid_y)
                all_data = [(xi, yi) for xi, yi in zip(train_x, train_y)]
                random.shuffle(all_data)
                x, y = zip(*all_data)
                train_x, train_y = x[:k], y[:k]
                valid_x, valid_y = x[k:], y[k:]
            if cls._train_on_all:
                train_x.extend(valid_x)
                train_y.extend(valid_y)
            train_datas.append(cls.TaskDataset(train_x, train_y))
            valid_datas.append(cls.TaskDataset(valid_x, valid_y))
        for train_data, valid_data, name in zip(train_datas,valid_datas,feddata['client_names']):
            train_data.optimal_model = valid_data.optimal_model = feddata[name]['optimal']
        return train_datas, valid_datas, test_data, feddata['client_names']

