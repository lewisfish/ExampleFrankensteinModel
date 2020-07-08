import argparse

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils.data import TensorDataset, DataLoader

import datasets
import engine
import models


parser = argparse.ArgumentParser(description='Mixed data ML model.')
parser.add_argument("-t", "--train", action="store_true", help="If flag is given then train all models, MLP, CNN, and CNNMLP")
parser.add_argument("-tc", "--traincnn", action="store_true", help="If flag is given then train CNN model")
parser.add_argument("-tm", "--trainmlp", action="store_true", help="If flag is given then train MLP model")
parser.add_argument("-tcm", "--trainmlpcnn", action="store_true", help="If flag is given then train CNNMLP model")
parser.add_argument("-e", "--evaluate", action="store_true", help="If flag is given then evaluate model")

args = parser.parse_args()

torch.manual_seed(0)

house_dataset = datasets.HouseDataset("Houses-dataset/", "HousesInfo.txt")
train_set, test_set = torch.utils.data.random_split(house_dataset, [272, 90])

trainLoader = DataLoader(train_set, batch_size=8, shuffle=False, num_workers=2, drop_last=True)
testLoader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1, drop_last=True)


if args.traincnn:
    CNNmodel = models.CNN()
    opt = optim.Adam(params=model.parameters(), lr=1e-3)
    lrscheduler = optim.lr_scheduler.ExponentialLR(optimizer=opt, gamma=0.96)
    loss_fn = nn.MSELoss()
    engine.trainCNN(CNNmodel, opt, loss_fn, trainLoader)

if args.trainmlp:
    MLPmodel = models.MLP(10)
    opt = optim.Adam(params=model.parameters(), lr=1e-3)
    lrscheduler = optim.lr_scheduler.ExponentialLR(optimizer=opt, gamma=0.96)
    loss_fn = nn.MSELoss()
    engine.trainMLP(MLPmodel, opt, loss_fn, trainLoader)

if args.trainmlpcnn:
    model = models.Frankenstein(10)
    opt = optim.Adam(params=model.parameters(), lr=1e-3)
    lrscheduler = optim.lr_scheduler.ExponentialLR(optimizer=opt, gamma=0.96)
    loss_fn = nn.MSELoss()
    engine.trainCNNMLP(CNNMLPmodel, opt, loss_fn, trainLoader)

if args.train:
    CNNmodel = models.CNN()
    CNNmodel.out = nn.Linear(16, 1)
    CNNmodel.act = nn.Sigmoid()

    MLPmodel = models.MLP(10)
    MLPmodel.out = nn.Linear(8, 1)
    MLPmodel.act = nn.Sigmoid()

    CNNMLPmodel = models.Frankenstein(10)

    opt = optim.Adam(params=CNNmodel.parameters(), lr=1e-3)
    lrscheduler = optim.lr_scheduler.ExponentialLR(optimizer=opt, gamma=0.96)
    loss_fn = nn.MSELoss()
    engine.trainCNN(CNNmodel, opt, loss_fn, trainLoader)
    print("Done CNN")

    opt = optim.Adam(params=MLPmodel.parameters(), lr=1e-3)
    lrscheduler = optim.lr_scheduler.ExponentialLR(optimizer=opt, gamma=0.96)
    loss_fn = nn.MSELoss()
    engine.trainMLP(MLPmodel, opt, loss_fn, trainLoader)
    print("Done MLP")

    opt = optim.Adam(params=CNNMLPmodel.parameters(), lr=1e-3)
    lrscheduler = optim.lr_scheduler.ExponentialLR(optimizer=opt, gamma=0.96)
    loss_fn = nn.MSELoss()
    engine.trainCNNMLP(CNNMLPmodel, opt, loss_fn, trainLoader)
    print("Done CNNMLP")

if args.evaluate:
    CNNmodel = models.CNN()
    # chop end of model and replace
    CNNmodel.out = nn.Linear(16, 1)
    CNNmodel.act = nn.Sigmoid()

    state = torch.load("CNN.pth")
    CNNmodel.load_state_dict(state["state_dict"])
    CNNmodel.eval()

    MLPmodel = models.MLP(10)
    # chop end of model and replace
    MLPmodel.out = nn.Linear(8, 1)
    MLPmodel.act = nn.Sigmoid()

    state = torch.load("MLP.pth")
    MLPmodel.load_state_dict(state["state_dict"])
    MLPmodel.eval()

    CNNMLPmodel = models.Frankenstein(10)
    state = torch.load("CNNMLP.pth")
    CNNMLPmodel.load_state_dict(state["state_dict"])
    CNNMLPmodel.eval()

    engine.evaluate(CNNmodel, MLPmodel, CNNMLPmodel, testLoader)
