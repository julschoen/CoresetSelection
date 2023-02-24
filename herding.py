import torch
import torch.nn.functional as F
import torchvision.models as models
from torchvision import datasets, transforms
import torchvision
import os
import argparse

def pretrain(model, args):
    if os.path.isfile('res.pt'):
        model.load_state_dict(storch.load('res.pt'))
        return model

    criterion = nn.CrossEntropyLoss().to(args.device)
    criterion.__init__()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5),
        transforms.Resize(224)
        ])

    dataset = datasets.CIFAR10('../data/', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1, batch_size=512, shuffle=True)


    for epoch in range(args.epochs):
        for i, (x,y) in enumerate(train_loader):
            x, y = x.to(args.device), y.to(args.device)
            opt.zero_grad()
            outputs = model(x)
            loss = self.criterion(outputs, y)
            loss = loss.mean()

            loss.backward()
            self.model_optimizer.step()

        acc = outputs.eq(y.view_as(outputs)).sum().item()

        print(f'[{epoch}|{args.epochs}] Loss {loss.item()}, Acc {acc.item()}')
    torch.save(model.state_dict(), 'res.pt')   
    return model

def log_interpolation(data, args):
    torchvision.utils.save_image(
        torchvision.utils.make_grid(data, nrow=args.num_ims, padding=2, normalize=True)
        , os.path.join(args.log_dir, f'ims.png'))



def herding_resnet():
    parser = argparse.ArgumentParser(description='Herding Arguments')
    # General
    parser.add_argument('--num_ims', type=int, default=10)
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--log_dir', type=str, default='../comparison_synth')
    parser.add_argument('--save_name', type=str, default='herding_1')

    # Pretrain
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)

    args = parser.parse_args()

    if not os.path.isdir(args.log_dir):
        os.mkdir(args.log_dir)

    # Load a pre-trained ResNet model
    resnet = models.resnet18(pretrained=True).to(args.device)
    resnet = pretrain(model, args)
    # Remove the last layer of the ResNet model
    resnet = torch.nn.Sequential(*(list(resnet.children())[:-1])).eval()

    with torch.no_grad():
        S = torch.zeros((args.num_classes*args.num_ims, 3, 32, 32), dtype=X.dtype, device=args.device)
        for c in range(args.num_classes):
            X = torch.load(os.path.join('../data/', f'data_class_{c}.pt'))
            # Extract features from the dataset using the ResNet model
            X_features = resnet(X)

            # Compute the empirical mean of the dataset
            mu = X_features.mean(dim=0)

            # Initialize the set of selected points and the set of unselected points
            U = X.clone()

            # Iteratively select the coreset points
            for i in range(args.num_ims):
                # Extract features from the unselected points using the ResNet model
                U_features = resnet(U)

                # Compute the similarity between the unselected points and the empirical mean based on the features
                sim = F.cosine_similarity(U_features, mu.view(1, -1), dim=1)

                # Find the index of the unselected point with the highest similarity
                j = torch.argmax(sim)

                # Add the selected point to the coreset and remove it from the set of unselected points
                S[(args.num_ims*c)+i] = U[j]
                U = torch.cat((U[:j], U[j+1:]))

                # Update the empirical mean based on the selected points
                mu = resnet(S[:(args.num_ims*c)+i+1].mean(dim=0).unsqueeze(0)).squeeze(0)
    log_interpolation(S, args)
    

if __name__ == '__main__':
    herding_resnet()