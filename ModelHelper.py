from cgi import test
from os import stat
from numpy.core.fromnumeric import prod
import torch
from torch import nn
from torch._C import StringType, device
from torch.nn.modules import loss 
from torch.utils.data import DataLoader, Dataset
from hyperopt import fmin, tpe, hp, Trials

from tqdm.notebook import tqdm
import pylab as plt

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



class DatasetRandom(Dataset):

    def __init__(self, size = 10000, shape = (20,20) ):
        """This generates n copies of 2D array in shape specified

        Args:
            size (tuple) : number of copies, in even number
            shape (tuple): shape of each copy of 2D array

        Returns:
            None
        """
 
        shape = (size//2,)+ shape 
        x1 = torch.normal(0, 2, size = shape) # y = 0
        x2 = torch.normal(0.2, 2, size = shape) # y = 1
        self.X = torch.cat([x1, x2], 0)

        y1 = torch.zeros(size//2)
        y2 = torch.ones(size//2)
        self.Y = torch.cat([y1, y2])

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        x = self.X[index]
        y = self.Y[index]

        return x, y



class myNeural(nn.Module):

    def __init__(self, 
                input_shape: tuple = (20,20),
                hidden_size: int = 256,
                act_func_1: torch.nn = nn.ReLU,
                act_func_2: torch.nn = nn.ReLU,
                config: dict = None):

        """myNeural is a neural network with 1 dense hidden layer

        Args:
            input_shape (tuple, optional)   : a tuple of shape for *each* instance of input, 
                                            in general, just input input_shape = X.shape[1:]. 
                                            Defaults to (20, 20)
            hidden_size (int, optional)     : size of input in the hidden layer.
                                            Defaults to 256
            act_func_1 (torch.nn, optional): activation function between input and hidden layer
                                            Defaults to nn.ReLU
            act_func_2 (torch.nn, optional): activation function between hidden and output layer
                                            Defaults to nn.ReLU
        """
        self.input_shape = input_shape
        self.hidden_size = hidden_size
        self.act_func_1 = act_func_1
        self.act_func_2 = act_func_2

        # if there is a config file given
        # config file will take precendence and overwrite
        # the other specification
        if config:
            for key, item in config.items():
                # only over-write those that are found in this class
                # so that the config file does not need to be complete
                # partial config can be specify by input to the constructor
                if key in self.__dict__:
                    self.__dict__[key] = item

        super(myNeural, self).__init__()

        input_size = reduce(mul, self.input_shape)
        # layers
        self.flatten = nn.Flatten()
        self.stack = nn.Sequential(
            nn.Linear(input_size, self.hidden_size),
            self.act_func_1(),
            nn.Linear(self.hidden_size, self.hidden_size),
            self.act_func_2(),
            nn.Linear(self.hidden_size, 1),
        )

        # every model comes with their individual trainer
        self.trainer = Trainer

    def forward(self, features):
        features = self.flatten(features)
        output = self.stack(features)
        #output = torch.sigmoid(output)
        output = torch.squeeze(output)
        return output

    def trainModel(self, 
                trainDataset, testDataset,
                lossFunction = nn.BCEWithLogitsLoss, 
                optimiser = torch.optim.SGD, lr = 0.1,
                epoch = 2, batch_size = 128,
                config: dict = None):
        """A wrapper function for the Trainer
        This training module will construct a Trainer using the config
        and call the training according to the config

        Args:
            trainDataset ([type]): [description]
            testDataset ([type]): [description]
            lossFunction ([type], optional): [description]. Defaults to nn.BCEWithLogitsLoss.
            optimiser ([type], optional): [description]. Defaults to torch.optim.SGD.
            lr (float, optional): [description]. Defaults to 0.1.
            epoch (int, optional): [description]. Defaults to 2.
            batch_size (int, optional): [description]. Defaults to 128.
            config (dict, optional): [description]. Defaults to None.
        """
        # can be wrapped into a function
        trainingConfig = {
            'lossFunction': lossFunction,
            'optimiser': optimiser, 'lr': lr,
            'epoch': epoch, 'batch_size': batch_size
        }
        if config:
            for var, value in config.items():
                if var in trainingConfig:
                    # over-write the variables
                    trainingConfig[var] = value

        # trainer should be able to take in a config dict also
        # need to change to function def
        self.trainer = self.trainer(self, trainDataset, testDataset,
                                    trainingConfig['lossFunction'], 
                                    trainingConfig['optimiser'], 
                                    trainingConfig['lr'])
        # print(type(int(trainingConfig['epoch'])), type(int(trainingConfig['batch_size'])))
        a = int(trainingConfig['epoch'])
        b = int(trainingConfig['batch_size'])
        # self.trainer.training(trainingConfig['epoch'], trainingConfig['batch_size'])
        self.trainer.training(a, b)

    def getTrainStats(self, stat = 'AvgTrainLoss'):
        return self.trainer.getTrainStats(stat)



class Trainer():
    def __init__(self, 
                model, 
                # x_train, y_train, x_test, y_test,
                trainDataset, testDataset,
                lossFunction = nn.BCEWithLogitsLoss, 
                optimiser = torch.optim.SGD, lr = 0.1):
        """[summary]

        Args:
            model (nn.Module)       : User defined Models
            trainDataset (Dataset)  : Trainning Data
            testDataset (Dataset)   : Test Data, recommended to be not too big
            lossFunction ([type], optional) : [description]. Defaults to nn.BCEWithLogitsLoss.
            optimiser ([type], optional)    : [description]. Defaults to torch.optim.SGD.
            lr (float, optional)            : [description]. Defaults to 0.1.
        """

        self.model = model
        self.trainDataset = trainDataset
        self.testDataset = testDataset

        self.lossFunction = lossFunction()
        self.learningRate = lr

        self.optimiser = optimiser(model.parameters(), lr = self.learningRate)

        self.modelHistory = {
            'trainLoss': [],
            'testLoss': [],
            'trainAcc': [],
            'testAcc': [],
        }
        self.trainingStats = dict()

    def training(self, epoch = 2, batch_size = 128):
        """[summary]

        Args:
            epoch (int, optional)     : [description]. Defaults to 2.
            batch_size (int, optional): [description]. Defaults to 128.
        """

        device = "cuda" if torch.cuda.is_available() else "cpu"
        trainLoader = DataLoader(self.trainDataset, batch_size= batch_size, shuffle=True)

        testX, testY = self.testDataset.X, self.testDataset.Y

        for epoch in tqdm(range(epoch), desc = 'Epoch'):

            per_epoch_bar = tqdm(enumerate(trainLoader), desc = 'Batch', leave = False)
            for batch, (x, y) in per_epoch_bar:
                x, y = x.to(device), y.to(device)
                
                # set to training mode
                self.model.train()
                # forward
                self.optimiser.zero_grad()
                pred = self.model(x)
                # print(pred.requires_grad)
                # why need to mannually turn it to True?
                # RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
                loss = self.lossFunction(pred, y)

                # backward
                loss.backward() # compute gradient
                self.optimiser.step() # move by lr

                pred = (pred>0).float()
                trainacc = (pred == y).sum()/len(y)
                # record train values
                self.modelHistory['trainLoss'].append(loss)
                self.modelHistory['trainAcc'].append(trainacc)


                # test loss
                # set to evaluation mode
                self.model.eval()
                with torch.no_grad():
                    testPred = self.model(testX)
                    testLoss = self.lossFunction(testPred, testY)
                testPred = (testPred>0).float()
                testAcc = (testPred == testY).sum()/len(testY)
                self.modelHistory['testLoss'].append(testLoss)
                self.modelHistory['testAcc'].append(testAcc)

                desc = f"Batch {batch} -- TrainLoss: {loss:0.5} -- TrainAcc: {trainacc:0.5%} -- TestAcc: {testAcc:0.5%}"

                if batch%20 == 0:
                    per_epoch_bar.set_description(desc)
                # print(desc)
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
        assert len(self.modelHistory['trainLoss']) > 0, "Training not conducted!"

        loss = self.modelHistory['trainLoss']
        self.trainingStats['AvgTrainLoss'] = sum(loss)/len(loss)
        self.trainingStats['MaxTrainLoss'] = max(loss)

        # record AvgTestAcc
        acc = self.modelHistory['testAcc']
        self.trainingStats['AvgTestAcc'] = sum(acc)/len(acc)
        self.trainingStats['negAvgTestAcc'] = -sum(acc)/len(acc)

        # regulrise if the deviation from the TrainAcc is too high
        # ignore the first 100 computations
        # after 100 iterations, the peformance for Test/Train should ideally be similar
        dev, c = 0, 0
        accTrain = self.modelHistory['trainAcc']
        accTest = self.modelHistory['testAcc']
        for i in range(0, len(acc)):
            dev += abs(accTrain[i] - accTest[i])
            c += 1
        self.trainingStats['negRegAvgTestAcc'] = self.trainingStats['negAvgTestAcc'] + 3*dev/c


    def plotTraining(self):
        plt.figure(figsize=(10,5))

        plt.subplot(1,2,1)
        plt.plot(self.modelHistory['trainLoss'], "-", c = 'dodgerblue', lw=3, alpha = 0.3, label="TRAIN")
        plt.plot(self.modelHistory['testLoss'], "-", c = 'red', lw=3, label="TEST")
        #plt.plot(loss_history, "-^")
        plt.xlabel("Iterations")
        plt.ylabel("loss")
        # plt.yscale("log")
        plt.title("Loss Function")
        plt.grid(True)

        plt.subplot(1,2,2)
        plt.plot(self.modelHistory['trainAcc'], "-", c = 'dodgerblue', lw=3, alpha = 0.3, label="TRAIN")
        plt.plot(self.modelHistory['testAcc'], "-", c = 'red', lw=2, label="TEST")
        plt.xlabel("Iterations")
        plt.ylabel("accuracy (%)")
        plt.legend()
        plt.title("Accuracy")
        plt.grid(True)
        plt.ylim(0,1)

    def getTrainStats(self, stat = 'AvgTrainLoss'):
        """return the training statistics from the Training Process
        """
        # assert stat in self.trainingStats, "Stats not computed"
        return self.trainingStats[stat]



class HyperTuner():

    def __init__(self, 
                trainDataset, testDataset,
                model = myNeural) -> None:
        self.model = model
        self.trainDataset = trainDataset
        self.testDataset = testDataset
        self.trials = Trials()


    def objGenerator(self, metric: str = 'AvgTrainLoss'):
        logger = myLoggers.HyperTuner_objGenerator
        logger.info(f'Objective function using {metric}')
        def objective(params: dict) -> float:
            """[summary]
            the objective function takes in the config,
            train the model according to the config then output the 
            AvgTrainLoss for evaluation

            Args:
                params (dict): configuration for the entire model training process

            Returns:
                float : Average Training Loss value
            """
            # initialise using the parameters
            model = self.model(config = params)
            # print(model.training)
            model.trainModel(self.trainDataset, self.testDataset, config = params)
            objValue = float(model.getTrainStats(stat = metric))

            logger = myLoggers.HyperTuner_objective
            logger.info(f'ObjValue={objValue:.10f} -- Params={params}')

            return objValue
        
        self.objective = objective

    def tunning(self, 
                paramSpace: dict, 
                metric: str = 'negRegAvgTestAcc', 
                algo = tpe.suggest,
                max_evals = 200
                ):

        """[summary]

        Metric:
            AvgTrainLoss
            MaxTrainLoss
            negAvgTestAcc
            negRegAvgTestAcc: Test Accuracy, regularised by penalising if the deviation from 
                              Train result is too large

        Returns:
            best parameters from the search
        """
        self.objGenerator(metric=metric)
        self.paramSpace = paramSpace

        best = fmin(
            fn=self.objective,
            space=paramSpace,
            algo=algo,
            max_evals=max_evals,
            trials=self.trials, # record the current performance
        )

        self.bestParam = best

        return best

    def tunningRay(self, paramSpace: dict,
                   metric:str = 'negRegAvgTestAcc'):
        """Conduct tunning using RayTune module

        Args:
            paramSpace (dict): dictionary of parameter space
            metric (str, optional): [description]. Defaults to 'negRegAvgTestAcc'.
        """
        pass


        












                