import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms 

from pyhessian import hessian 

def train(num_epochs,model,dataloader,criterion,optimizer,max_iteration=1000000):
    for epoch in range(num_epochs):
        for i, (inputs, targets) in enumerate(tqdm(dataloader)):
            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs,targets)
            loss.backward()
            optimizer.step()

            if i % 1000 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
#                fp.write(f'Epoch,{epoch+1}, {loss.item():.4f}')
                
            if(i>max_iteration):
                break
    return model

def calcHessianTrace(model,criterion,dataloader=None,data=None,cuda=False):
    comp=hessian(model, criterion, dataloader=dataloader,data=data,cuda=cuda)
    trace=comp.trace()
    return trace

def main(args):
    num_epochs=args.num_epochs
    batchsize=args.batchsize
    lr=args.learningrate
    data_dir=args.data_dir
    max_iteration=args.max_iteration    
    pretrained=args.pretrained
    modelname=args.modelname
    DEBUG=args.DEBUG

    tr=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    if(args.dataset=="CIFAR10"):
        dataset_train = datasets.CIFAR10(data_dir, train=True, download=True,transform=tr)
        label_num=10
    elif(args.dataset=="CIFAR100"):
        dataset_train = datasets.CIFAR100(data_dir, train=True, download=True,transform=tr)
        label_num=100
    elif(args.dataset=="FashionMNIST"):
        dataset_train = datasets.FashionMNIST(data_dir, train=True, download=True,transform=tr)
        label_num=10
    else:
        tr=transforms.Compose([
            transforms.Resize((7,7)), #resnet18
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3,1,1)),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        dataset_train = datasets.MNIST(data_dir, train=True, download=True,transform=tr)
        label_num=10

    #from pytorchcv.model_provider import get_model as ptcv_get_model 
    #quantized_net = resnet20_cifar()

    dataloader = DataLoader(dataset_train, batchsize, shuffle=True)
    #criterion = LabelSmoothingCrossEntropy(args.smoothing) 
    criterion = nn.CrossEntropyLoss()

    if(modelname=="mobilenet"):
        model= torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=pretrained)
    elif(modelname=="resnet50"):
        model= torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=pretrained)    
    else:
        model= torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=pretrained)

    if(args.opt=="Adam"):
        optimizer = optim.Adam(list(model.parameters()),  lr = lr)
    else:
        optimizer = optim.SGD(model.parameters(), lr=lr)

    if(not args.pretrained):
        model=train(num_epochs,model,dataloader,criterion,optimizer,max_iteration)

    if(DEBUG):
        data=next(iter(dataloader))
        trace=calcHessianTrace(model,criterion,dataloader=None,data=data,cuda=args.cuda)    
        print(trace)
        print("len",len(trace))
    else:
        trace=calcHessianTrace(model,criterion,dataloader,cuda=args.cuda)    

    #verification

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,default='data', help='Directory for storing input data')
    parser.add_argument('-n', '--num_epochs',default=10,type=int)
    parser.add_argument('-bs', '--batchsize',default=4,type=int)
    parser.add_argument('-ds', '--dataset',default="mnist")    
    parser.add_argument('-l', '--learningrate',default=1e-4,type=float)
    parser.add_argument('-opt', '--opt',default="SGD")    
    parser.add_argument('-m', '--max_iteration',default=1000000,type=int)
    parser.add_argument('-p', '--pretrained',default=True)
    parser.add_argument('-D', '--DEBUG',default=False)
    parser.add_argument('-c', '--cuda',default=False)
    parser.add_argument('-mn', '--modelname',default="")

    FLAGS, unparsed = parser.parse_known_args()
    args = parser.parse_args()
    main(args)