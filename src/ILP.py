import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms 

import torch.utils.data as data
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.backends.cudnn as cudnn
import torch.optim.lr_scheduler as lr_scheduler
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from pyhessian import hessian 
import numpy as np
import pulp
import LayerHessian as L

scale=lambda w,m,M:(w-m)/(M-m)
unscale=lambda w,m,M:(M-m)*w+m

def quantize(ws,bitwidth,m,M):
    bb=(1<<bitwidth)
    ws=scale(ws,m,M)*bb
    ws=ws.to(torch.int64).to(torch.float64)/bb
    ws=unscale(ws,m,M)
    return ws

quantize_local= lambda ws,b:quantize(ws,b,torch.min(ws),torch.max(ws))

def calc_delta_weight_sq(model,bitwidth):
    w=list(model.parameters())
    deltaw=[ quantize_local(wi,bitwidth)-wi for wi in w]
    return deltaw*deltaw

deltaw=[ [quantize_local(wi,b)-wi for b in Bset] for wi in w]

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

#from LayerHessian
def calcHessianTrace_layerwise(model,criterion,dataloader=None,data=None,cuda=False):

    scheduler = lr_scheduler.MultiStepLR(optimizer, args.milestones, gamma=args.step_gamma)    

    for layer_name, _ in model.named_parameters():
        print('layer:', layer_name)

    model.train()

    full_eigenspectrums = list()
    epoch_eigenspectrums = list()
    full_eigenspectrums_path = osp.join(ckpt_dir, 'training_eigenspectrum_full.npy')
    
    C = config.num_classes
    valid_layers = get_valid_layers(model)
    for epoch in range(num_epochs):
        logger.info('epoch: %d' % epoch)
        with torch.enable_grad():
            for batch, truth in dataloaders['train']:

                batch = batch.to(device)
                truth = truth.to(device)
                optimizer.zero_grad()

                output = model(batch)
                loss = criterion(output, truth)

                loss.backward()
                optimizer.step()

        scheduler.step()

        # updates finished for epochs
        mean, std = get_mean_std(args.dataset)
        pad = int((config.padded_im_size-config.im_size)/2)
        transform = transforms.Compose([transforms.Pad(pad),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean,std)])
        if args.dataset in ['MNIST', 'FashionMNIST', 'CIFAR10', 'CIFAR100']:
            full_dataset = getattr(datasets, args.dataset)
            subset_dataset = get_subset_dataset(full_dataset=full_dataset,
                                                examples_per_class=args.examples_per_class,
                                                epc_seed=config.epc_seed,
                                                root=osp.join(args.dataset_root, args.dataset),
                                                train=True,
                                                transform=transform,
                                                download=True
                                                )
        elif args.dataset in ['STL10', 'SVHN']:
            full_dataset = getattr(datasets, args.dataset)
            subset_dataset = get_subset_dataset(full_dataset=full_dataset,
                                                examples_per_class=args.examples_per_class,
                                                epc_seed=config.epc_seed,
                                                root=osp.join(args.dataset_root, args.dataset),
                                                split='train',
                                                transform=transform,
                                                download=True
                                                )
        else:
            raise Exception('Unknown dataset: {}'.format(args.dataset))

        loader = data.DataLoader(dataset=subset_dataset,
                            drop_last=False,
                            batch_size=args.batch_size)
        
        Hess = L.FullHessian(crit='CrossEntropyLoss',
                            loader=loader,
                            device=device,
                            model=model,
                            num_classes=C,
                            hessian_type='Hessian',
                            init_poly_deg=64,
                            poly_deg=128,
                            spectrum_margin=0.05,
                            poly_points=1024,
                            SSI_iters=128
                            )

        Hess_eigval, Hess_eigval_density = Hess.LanczosLoop(denormalize=True)

        full_eigenspectrums.append(Hess_eigval)
        full_eigenspectrums.append(Hess_eigval_density)


        for layer_name, _ in model.named_parameters():
            if layer_name not in valid_layers:
                continue
                
            Hess = L.LayerHessian(crit='CrossEntropyLoss',
                                loader=loader,
                                device=device,
                                model=model,
                                num_classes=C,
                                layer_name=layer_name,
                                hessian_type='Hessian',
                                init_poly_deg=64,
                                poly_deg=128,
                                spectrum_margin=0.05,
                                poly_points=1024,
                                SSI_iters=128
                                )

            Hess_eigval, \
            Hess_eigval_density = Hess.LanczosLoop(denormalize=True)

            layerwise_eigenspectrums_path = osp.join(ckpt_dir, 'training_eigenspectrums_epoch_{}_layer_{}.npz'.format(epoch, layer_name))
            np.savez(layerwise_eigenspectrums_path, eigval=Hess_eigval, eigval_density=Hess_eigval_density)



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
        tarce=np.array(trace)
        np.savetxt("trace.csv",trace)

    #verification
   
def ILP(Hutchinson_trace,model,bits):
    delta_weights_sq= [for b in bits]
    deltaw2_8bit=calc_delta_weight_sq(model,8)
    deltaw2_4bit=calc_delta_weight_sq(model,4)
    delta_weights_dif=deltaw2_8bit-deltaw2_4bit
    
    paramnum=getparamnum(model)
    bops =getBOPS(model)

    #resnet constraint
#    latency=get

def ILP_2canditates(trace,model,canditatebits,with_size_limit with_ops_limit,with_latency_limit):
    delta_weights=[ calc_delta_weight_sq(model,8), calc_delta_weight_sq(model,4)]

#parameters from https://github.com/Zhen-Dong/HAWQ/blob/main/ILP.ipynb
def ILP_sample_Resnet18_8_4(model,with_size_limit with_ops_limit,with_latency_limit):
    Hutchinson_trace=np.array(trace)
#    delta_weights_8bit_square = np.array([0.0235, 0.0125, 0.0102, 0.0082, 0.0145, 0.0344, 0.0023, 0.0287, 0.0148, 0.0333, 0.0682, 0.0027, 0.0448, 0.0336, 0.0576, 0.1130, 0.0102, 0.0947, 0.0532]) #  = (w_fp32 - w_int8)^2
    # Delta Weight 4 bit Square means \| W_fp32 - W_int4  \|_2^2
#    delta_weights_4bit_square = np.array([6.7430, 3.9691, 3.3281, 2.6796, 4.7277, 10.5966, 0.6827, 9.0942, 4.8857, 10.7599, 21.7546, 0.8603, 14.5324, 10.9651, 18.7706, 36.4044, 3.1572, 29.6994, 17.4016]) #  = (w_fp32 - w_int4)^2
    delta_weights=[ calc_delta_weight_sq(model,8), calc_delta_weight_sq(model,4)]
    delta_weights_dif=delta_weights[0]-delta_weights[1]
    
    # number of paramers of each layer
    paramnum= np.array([ 36864, 36864, 36864, 36864, 73728, 147456, 8192, 147456, 147456, 294912, 589824, 32768, 589824, 589824, 1179648, 2359296, 131072, 2359296, 2359296]) / 1024 / 1024 # make it millition based (1024 is due to Byte to MB see next cell for model size computation)
    # Bit Operators of each layer
    bops = np.array([115605504, 115605504, 115605504, 115605504, 57802752, 115605504, 6422528, 115605504, 115605504, 57802752, 115605504, 6422528, 115605504, 115605504, 57802752, 115605504, 6422528, 115605504, 115605504]) / 1000000 # num of bops for each layer/block

    # latency for INT4 and INT8, measured on T4 GPU
    latency_int4 = np.array([ 0.21094404, 0.21092674, 0.21104113, 0.21086851, 0.13642465, 0.19167506, 0.02532183, 0.19148203, 0.19142914, 0.11395316, 0.20556011, 0.01917474, 0.20566918, 0.20566509, 0.13185102, 0.22287786, 0.01790088, 0.22304611, 0.22286099])
    latency_int8 = np.array([ 0.36189111, 0.36211718, 0.31141909, 0.30454471, 0.19184896, 0.38948934, 0.0334169, 0.38904905, 0.3892859, 0.19134735, 0.34307431, 0.02802354, 0.34313329, 0.34310756, 0.21117103, 0.37376585, 0.02896843, 0.37398187, 0.37405185])

    # model size
    model_size_32bit = np.sum(paramnum) * 4. # MB
    model_size_8bit = model_size_32bit / 4. # 8bit model is 1/4 of 32bit model 
    model_size_4bit = model_size_32bit / 8. # 4bit model is 1/8 of 32bit model 
    # as mentioned previous, that's how we set the model size limit
    model_size_limit = model_size_4bit + (model_size_8bit - model_size_4bit) * args.model_size_limit

    # bops
    bops_8bit = bops / 4. / 4. # For Wx, we have two matrices, so that we need the (bops / 4 / 4)
    bops_4bit = bops / 8. / 8. # Similar to above line
    bops_limit = np.sum(bops_4bit) + (np.sum(bops_8bit) - np.sum(bops_4bit)) * args.bops_limit # similar to model size
    
    bops_dif_4_8= bops_8bit - bops_4bit
    # latency 
    latency_limit = np.sum(latency_int4) + (np.sum(latency_int8) - np.sum(latency_int4)) * args.latency_limit # similar to model size
    latency_dif_4_8= latency_int8- latency_int4

    # first get the variables, here 1 means 4 bit and 2 means 8 bit
    num_variable = Hutchinson_trace.shape[0]
    variable = {variable[f"x{i}"]:pulp.LpVariable(f"x{i}", 1, 2, cat=pulp.LpInteger) for i in range(num_variable)}
    
    prob = pulp.LpProblem("Model_Size", pulp.LpMinimize)
    prob += sum([0.5 * variable[f"x{i}"] * paramnum[i] for i in range(num_variable) ]) <= model_size_limit # 1 million 8 bit numbers means 1 Mb, here 0.5 * 2 = 1, similar for 4 bit

    prob += sum([ bops_4bit[i] + (variable[f"x{i}"] - 1) * bops_dif_4_8[i] for i in range(num_variable) ]) <= bops_limit
    prob += sum([ latency_int4[i] + (variable[f"x{i}"] - 1) * latency_dif_4_8[i] for i in range(num_variable) ]) <= latency_limit # if 4 bit, x - 1 = 0, we use bops_4, if 8 bit, x-1=2, we use bops_diff + bos_4 = bops_8
    # add downsampling layer constraint, here we make the residual connetction
    # layer have the same bit as the main stream
    prob += variable[f"x4"] ==variable[f"x6"]
    prob += variable[f"x9"] ==variable[f"x11"]
    prob += variable[f"x14"] ==variable[f"x16"]

    sensdif= Hutchinson_trace * delta_weights_dif

    # here is the sensitivity different between 4 and 8

    # for fixed bops, we want large models, as well as smaller sensitivy
    # here sensitivity_difference_between is negative numbers, so if x = 1, means using 4 bit, gives us 0, and if x = 2, means using 8 bits, gives us negavie numers. It will prefer 8 bit
    # negative model size is negaive number. It will prefer 8 bit.
    # both prefer 8 bit but the bops is constrained, so we get a tradeoff. 
    prob += sum( [ (variable[f"x{i}"] - 1) * sensdif[i] for i in range(num_variable) ] )  

    # solve the problem
    status = prob.solve(pulp.GLPK_CMD(msg=1, options=["--tmlim", "10000","--simplex"]))
    #status = prob.solve(COIN_CMD(msg=1, options=['dualSimplex']))
    #status = prob.solve(GLPK(msg=1, options=["--tmlim", "10","--dualSimplex"]))

    # get the result
    pulp.LpStatus[status]

    result = [pulp.value(variable[f"x{i}"]) for i in range(num_variable)]
    return np.array(result)
    

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