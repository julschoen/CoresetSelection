import torch
import torch.nn.functional as F
import torchvision.models as models
from torchvision import datasets, transforms
import torchvision
import os
import argparse

def pretrain(model, args, run_num):
    if os.path.isfile(f'res_{run_num}.pt'):
        print('### Loading Pretrained ResNet-18 ###')
        model.load_state_dict(torch.load(f'res_{run_num}.pt'))
        return model

    criterion = torch.nn.CrossEntropyLoss().to(args.device)
    criterion.__init__()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5),
        transforms.Resize(224)
        ])

    dataset = datasets.CIFAR10('../data/', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=True)


    for epoch in range(args.epochs):
        for i, (x,y) in enumerate(train_loader):
            x, y = x.to(args.device), y.to(args.device)
            opt.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss = loss.mean()

            loss.backward()
            opt.step()

        pred = outputs.argmax(dim=1, keepdim=True)
        acc = (pred.eq(y.view_as(pred)).sum().item()/512.)*100.

        print(f'[{epoch}|{args.epochs}] Loss {loss.item()}, Acc {acc}')
    torch.save(model.state_dict(), f'res_{run_num}.pt')   
    return model

def log_interpolation(data, args):
    torchvision.utils.save_image(
        torchvision.utils.make_grid(data, nrow=args.num_ims, padding=2, normalize=True)
        , os.path.join(args.log_dir, f'ims.png'))

def save(file_name, data, comp_dir, num_ims):
        file_name = os.path.join(comp_dir, str(num_ims), file_name)
        torch.save(data.cpu(), file_name)
        
def herding_resnet():
    parser = argparse.ArgumentParser(description='Herding Arguments')
    # General
    parser.add_argument('--num_ims', type=int, default=10)
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--log_dir', type=str, default='../comparison_synth')

    # Pretrain
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-3)

    args = parser.parse_args()

    if not os.path.isdir(args.log_dir):
        os.mkdir(args.log_dir)

    for k in range(10):
        print(f'### Run {k+1} ###')
        resnet = models.resnet18(weights='DEFAULT')
        resnet.fc = torch.nn.Linear(512,10)
        resnet = resnet.to(args.device)
        resnet = pretrain(resnet, args, k)
        resnet = torch.nn.Sequential(*(list(resnet.children())[:-1])).eval().cpu().float()

        with torch.no_grad():
            S = torch.zeros((args.num_classes*args.num_ims, 3, 32, 32), dtype=torch.float)
            for c in range(args.num_classes):
                print(f'### Class {c} ###')
                X = torch.load(os.path.join('../data/', f'data_class_{c}.pt'))
                mu = resnet(X).mean(dim=0)

                # Initialize the set of unselected points
                U = X.clone()

                for i in range(args.num_ims):
                    U_features = resnet(U).squeeze()
                    sim = F.cosine_similarity(U_features, mu.view(1, -1), dim=1)
                    j = torch.argmax(sim)

                    # Add the selected point to the coreset and remove it from the set of unselected points
                    S[(args.num_ims*c)+i] = U[j]
                    U = torch.cat((U[:j], U[j+1:]))

                    # Update the empirical mean based on the selected points
                    #mu = resnet(S[:(args.num_ims*c)+i+1].mean(dim=0).unsqueeze(0)).squeeze(0)
                    mu = resnet(U).mean(dim=0)

        save(f'herding_x_{k}.pt', S, args.log_dir, args.num_ims)
        save(f'herding_y_{k}.pt', torch.arange(args.num_classes).repeat(args.num_ims,1).T.flatten(), args.log_dir, args.num_ims)
    

if __name__ == '__main__':
    herding_resnet()