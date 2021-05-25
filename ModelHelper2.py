# from cgi import test
# from os import stat

from numpy.core.fromnumeric import prod
from ray.tune import analysis, result

import torch
from torch import nn
from torch._C import StringType, device
from torch.nn.modules import loss 
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

from hyperopt import fmin, tpe, hp, Trials
from ray import tune
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune.schedulers import ASHAScheduler


from tqdm.notebook import tqdm
import pylab as plt

from collections import defaultdict
from functools import reduce
from operator import mul
import logging
import myLoggers

logging.basicConfig(
        level=logging.DEBUG, # DEBUG and above will be logged
        filename='logs/log',
        filemode='a',
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

loggerMyNeural = myLoggers.myNeural
loggermyNeuralHyperTunner = myLoggers.myNeuralHyperTunner

class myNeural(nn.Module, tune.Trainable):

    def __init__(self, 
                config: dict = {
                    'input_shape':(20,20),
                    'hidden_size': 256,
                    'act_func_1': nn.ReLU,
                    'act_func_2': nn.ReLU
                }, 
                 **kwargs):

        """myNeural is a neural network with 1 dense hidden layer

        Config (dict):
            input_shape (tuple, optional)   : a tuple of shape for *each* instance of input, 
                                            in general, just input input_shape = X.shape[1:]. 
                                            Defaults to (20, 20)
            hidden_size (int, optional)     : size of input in the hidden layer.
                                            Defaults to 256
            act_func_1 (torch.nn, optional): activation function between input and hidden layer
                                            Defaults to nn.ReLU
            act_func_2 (torch.nn, optional): activation function between hidden and output layer
                                            Defaults to nn.ReLU
        kwards: arguments that are present in the config, these keyword args will overwrite the
                config 
        """

        self.config = config
        # update the config if any kwargs are specified
        self.config.update((k, kwargs[k]) for k in config.keys() & kwargs.keys())

        super(myNeural, self).__init__()

        input_size = reduce(mul, self.config['input_shape'])
        # layers
        self.flatten = nn.Flatten()
        self.stack = nn.Sequential(
            nn.Linear(input_size, self.config['hidden_size']),
            self.config['act_func_1'](),
            nn.Linear(self.config['hidden_size'], self.config['hidden_size']),
            self.config['act_func_2'](),
            nn.Linear(self.config['hidden_size'], 1),
        )

        self.stats = defaultdict(list)

        # log the construction
        loggerMyNeural.info(f"New Model: {self.config}")

    def forward(self, features):

        features = self.flatten(features)
        output = self.stack(features)
        #output = torch.sigmoid(output)
        output = torch.squeeze(output)
        loggerMyNeural.info(f'Forward Pass: {output}')
        return output


##     def step(self):

    def optimise(self, 
                trainData: Dataset, testData: Dataset,
                config:dict = {
                    'lossFunction': nn.BCEWithLogitsLoss, 
                    'optimiser': torch.optim.SGD, 'lr': 0.1,
                    'epoch': 2, 'batch_size': 128
                },
                hideEpochProgressBar = True,
                hideBatchProgressBar = True):
        """This method optimise this model according to the parameter
        complete the entire optimisation process,
        process is tracked and recorded, may be terminated in advance if 
        RayTune ASHA scheduler is used

        Args:
            trainData (Dataset)                     : [description]
            testData (Dataset)                      : [description]
            config (dict, optional)                 : 
                                                    Must contain epoch and batch_size for config. 
                                                    Defaults to { 'epoch': 2, 'batch_size': 128 }.
            lossFunction ([type], optional)         : [description]. Defaults to nn.BCEWithLogitsLoss.
            optimiser ([type], optional)            : [description]. Defaults to torch.optim.SGD.
            lr (float, optional)                    : Learning Rate. Defaults to 0.1.
            hideEpochProgressBar                    : [description]
            hideBatchProgressBar                    : [description]
        """
        # device = "cuda" if torch.cuda.is_available() else "cpu"
        epoch = config['epoch']
        batch_size = config['batch_size']
        lossFunction = config['lossFunction']()
        lr = config['lr']
        optimiser = config['optimiser'](self.parameters(), lr = lr)
        
        trainLoader = DataLoader(trainData, 
                                batch_size= batch_size, 
                                shuffle=True)
        testX, testY = testData.X, testData.Y

        for epoch in tqdm(range(epoch), 
                            desc = 'Epoch', 
                            disable=hideEpochProgressBar,
                            leave=False):

            per_epoch_bar = tqdm(enumerate(trainLoader), 
                                desc = 'Batch', 
                                leave = False,
                                disable=hideBatchProgressBar)
            for batch, (x, y) in per_epoch_bar:
                #self.step()
                # x, y = x.to(device), y.to(device)

                # set to training mode
                self.train()
                optimiser.zero_grad()
                # "forward" and produce output
                pred = self.__call__(x)
                loss = lossFunction(pred, y)

                # backward
                loss.backward() # compute gradient
                optimiser.step() # move by lr

                pred = (pred>0).float()
                trainacc = (pred == y).sum()/len(y)

                # record train values
                self.stats['trainLoss'].append(loss)
                self.stats['trainAcc'].append(trainacc)

                # test loss
                # set to evaluation mode
                self.eval()
                with torch.no_grad():
                    testPred = self.__call__(testX)
                    testLoss = lossFunction(testPred, testY)

                testPred = (testPred>0).float()
                testAcc = (testPred == testY).sum()/len(testY)
                self.stats['testLoss'].append(testLoss)
                self.stats['testAcc'].append(testAcc)

                # tune tracking
                tune.report(
                    trainAcc=trainacc,
                    testAcc = testAcc,
                    # penalise the testAcc if the value is too different
                    testAccReg = testAcc -abs(trainacc - testAcc) 
                    )


                desc = f"Batch {batch} -- TrainLoss: {loss:0.5} -- TrainAcc: {trainacc:0.5%} -- TestAcc: {testAcc:0.5%}"
                if batch%20 == 0:
                    per_epoch_bar.set_description(desc)
                    loggerMyNeural.info(desc)
        
        # compute scores for this Training process
        self.summmariseTraining()

    def summmariseTraining(self) -> None:
        """[summary]
        Compute for summary statistics for the training to keep track of the 
        performance of the training.

        Metric:
            AvgTrainLoss
            MaxTrainLoss
            negAvgTestAcc    

        Returns:
            None
        """
        assert len(self.stats['trainLoss']) > 0, "Training not conducted!"

        loss = self.stats['trainLoss']
        self.stats['AvgTrainLoss'] = sum(loss)/len(loss)
        self.stats['MaxTrainLoss'] = max(loss)

        # record AvgTestAcc
        acc = self.stats['testAcc']
        self.stats['AvgTestAcc'] = sum(acc)/len(acc)
        self.stats['negAvgTestAcc'] = -sum(acc)/len(acc)

        # regulrise if the deviation from the TrainAcc is too high
        # ignore the first 100 computations
        # after 100 iterations, the peformance for Test/Train should ideally be similar
        dev, c = 0, 0
        accTrain = self.stats['trainAcc']
        accTest = self.stats['testAcc']
        for i in range(0, len(acc)):
            dev += abs(accTrain[i] - accTest[i])
            c += 1
        self.stats['negRegAvgTestAcc'] = self.stats['negAvgTestAcc'] + 3*dev/c

        loggerMyNeural.info(f"Training Done")

    def getTrainStats(self, stat = 'AvgTrainLoss'):
        assert stat in self.stats, "Statistics not found"
        return self.stats[stat]


    def plotTraining(self):
        plt.figure(figsize=(10,5))

        plt.subplot(1,2,1)
        plt.plot(self.stats['trainLoss'], "-", c = 'dodgerblue', lw=3, alpha = 0.3, label="TRAIN")
        plt.plot(self.stats['testLoss'], "-", c = 'red', lw=3, label="TEST")
        #plt.plot(loss_history, "-^")
        plt.xlabel("Iterations")
        plt.ylabel("loss")
        # plt.yscale("log")
        plt.title("Loss Function")
        plt.grid(True)

        plt.subplot(1,2,2)
        plt.plot(self.stats['trainAcc'], "-", c = 'dodgerblue', lw=3, alpha = 0.3, label="TRAIN")
        plt.plot(self.stats['testAcc'], "-", c = 'red', lw=2, label="TEST")
        plt.xlabel("Iterations")
        plt.ylabel("accuracy (%)")
        plt.legend()
        plt.title("Accuracy")
        plt.grid(True)
        plt.ylim(0,1)

    @classmethod
    def fitTrain(cls, config:dict) -> None:
        """FitTrain simulates the entire process
        of constructing a model and optimising it
        and in the process recording the performance metrics.
        This is a classmethod for hyperTuner, so that it is able
        to gather the best config

        Args:
            config (dict): [description]

        Returns:
            [type]: [description]
        """
        # construction
        model = cls(config)
        model.optimise(cls.trainData, cls.testData, config)

    @classmethod
    def hyperTuner(cls,
                    trainData: Dataset, testData: Dataset,
                    paramSpace:dict,
                    numSamples = 50,

                    searchAlg = HyperOptSearch,
                    searchMode = 'max',
                    searchMaxConcurrent = 2,
                    # searchRewardAttr = 'testAccReg',
                    
                    scheduler = ASHAScheduler,
                    schedulerMetric = 'testAccReg',
                    schedulerMode = 'max',
                    schedulerGracePeriod = 1) -> tune.Analysis:
        """hyperTuner uses RayTune to tune all the hyper-paramaters 
        defined in config, it calls upon fitTrain


        Args:
            trainData (Dataset): [description]
            testData (Dataset): [description]
            paramSpace (dict): [description]
            numSamples (int, optional): [description]. Defaults to 50.

            searchAlg ([type], optional): [description]. Defaults to HyperOptSearch.
            searchMode (str, optional): [description]. Defaults to 'max'.
            searchMaxConcurrent (int, optional): [description]. Defaults to 2.
            searchRewardAttr (str, optional): [description]. Defaults to 'testAccReg'.

            scheduler ([type], optional): ASHAScheduler is able to 
                                        perform early stopping.
                                        Defaults to ASHAScheduler.

            schedulerMetric (str, optional): [description]. Defaults to 'testAccReg'.
            schedulerMode (str, optional): [description]. Defaults to 'max'.
            schedulerGracePeriod (int, optional): [description]. Defaults to 1.

        Returns:
            tune.Analysis: [description]
        """              
        cls.trainData = trainData
        cls.testData = testData

        searchAlg = searchAlg(paramSpace, 
                                max_concurrent=searchMaxConcurrent, 
                                # reward_attr=searchRewardAttr,
                                mode=searchMode)
        
        scheduler = scheduler(metric = schedulerMetric,
                                mode = schedulerMode,
                                grace_period = schedulerGracePeriod)
        analysis = tune.run(cls.fitTrain,
                            num_samples=numSamples,
                            #search_alg=searchAlg,
                            scheduler=scheduler
        )
        cls.analysis = analysis
        return analysis

        

        









                