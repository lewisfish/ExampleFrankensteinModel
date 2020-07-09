import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import tqdm


def metric(preds, gts):

    ape = torch.tensor([0.0])
    for i, j in zip(preds, gts):
        a = i.flatten()
        ape += torch.abs((a - j) / a)
    return ape.item() * (100. / len(preds))


def evaluate(cnnmodel, mlpmodel, cnnmlpmodel, testLoader):

    # infer on CNNMLP model
    cnnmlpmodel.eval()

    outputs = []
    gts = []
    with torch.no_grad():
        for i, item in enumerate(tqdm.tqdm(testLoader)):
            images = item[0]

            datas = item[1]
            targets = item[2]

            output = cnnmlpmodel(images, datas)
            outputs.append(output)
            gts.append(targets)

    cnnmlp = metric(outputs, gts)

    # infer on CNN model
    cnnmodel.eval()

    outputs = []
    gts = []
    with torch.no_grad():
        for i, item in enumerate(tqdm.tqdm(testLoader)):
            images = item[0]

            targets = item[2]

            output = cnnmodel(images)
            outputs.append(output)
            gts.append(targets)

    cnn = metric(outputs, gts)

    # infer on MLP model
    mlpmodel.eval()

    outputs = []
    gts = []
    with torch.no_grad():
        for i, item in enumerate(tqdm.tqdm(testLoader)):

            datas = item[1]
            targets = item[2]

            output = mlpmodel(datas)
            outputs.append(output)
            gts.append(targets)

    mlp = metric(outputs, gts)

    print(f"MAPE CNN: {cnn:.2f}")
    print(f"MAPE MLP: {mlp:.2f}")
    print(f"MAPE CNNMLP: {cnnmlp:.2f}")


def trainMLP(model, optimiser, loss_fn, trainLoader, Numepochs=200):
    writer = SummaryWriter('runs/mlp')

    running_loss = 0.0
    for epoch in tqdm.tqdm(range(0, Numepochs)):
        for i, item in enumerate(trainLoader, 0):

            datas = item[1]
            targets = item[2]

            output = model(datas)
            output = torch.flatten(output)

            loss = loss_fn(output, targets)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            running_loss += loss.item()
            if i % 100:
                writer.add_scalar("training loss", running_loss / 100, epoch*len(trainLoader)+1)
                running_loss = 0.0

    writer.close()
    state = {"state_dict": model.state_dict(),
             "optimizer": optimiser.state_dict()}
    torch.save(state, "MLP.pth")


def trainCNN(model, optimiser, loss_fn, trainLoader, Numepochs=200):
    writer = SummaryWriter('runs/cnn')

    running_loss = 0.0
    for epoch in tqdm.tqdm(range(0, Numepochs)):
        for i, item in enumerate(trainLoader, 0):

            images = item[0]
            targets = item[2]

            output = model(images)
            output = torch.flatten(output)

            loss = loss_fn(output, targets)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            running_loss += loss.item()
            if i % 100:
                writer.add_scalar("training loss", running_loss / 100, epoch*len(trainLoader)+1)
                running_loss = 0.0

    writer.close()
    state = {"state_dict": model.state_dict(),
             "optimizer": optimiser.state_dict()}
    torch.save(state, "CNN.pth")


def trainCNNMLP(model, optimiser, loss_fn, trainLoader, Numepochs=200):
    writer = SummaryWriter('runs/cnnmlp')

    running_loss = 0.0
    for epoch in tqdm.tqdm(range(0, Numepochs)):
        for i, item in enumerate(trainLoader, 0):

            images = item[0]

            datas = item[1]
            targets = item[2]

            output = model(images, datas)
            output = torch.flatten(output)

            loss = loss_fn(output, targets)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            running_loss += loss.item()
            if i % 100:
                writer.add_scalar("training loss", running_loss / 100, epoch*len(trainLoader)+1)
                running_loss = 0.0

    writer.close()
    state = {"state_dict": model.state_dict(),
             "optimizer": optimiser.state_dict()}
    torch.save(state, "CNNMLP.pth")
