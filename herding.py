import torch
import torch.nn.functional as F
import torchvision.models as models
from torchvision import datasets, transforms

def pretrain(model, args):
    criterion = nn.CrossEntropyLoss().to(args.device)
    criterion.__init__()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5),
        transforms.Resize(224)
        ])

    dataset = datasets.CIFAR10('../data/', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)


    for epoch in range(args.epochs):
        for i, (x,y) in enumerate(train_loader):
            x, y = x.to(args.device), y.to(args.device)

            # Forward propagation, compute loss, get predictions
            opt.zero_grad()
            outputs = model(x)
            loss = self.criterion(outputs, y)
            loss = loss.mean()

            loss.backward()
            self.model_optimizer.step()

        acc = outputs.eq(y.view_as(outputs)).sum().item()

        print(f'[{epoch}|{args.epochs}] Loss {loss.item()}, Acc {acc.item()}')

    return model



def herding_resnet(X, m, args):
    """
    Selects a coreset of size m from the dataset X using the Herding algorithm and a pre-trained ResNet model.

    Args:
    - X: the dataset, a PyTorch tensor of size (N, C, H, W) where N is the number of images, C is the number of channels, 
         and H and W are the height and width of each image, respectively.
    - m: the size of the coreset to select.

    Returns:
    - A PyTorch tensor of size (m, C, H, W) containing the selected coreset.
    """
    N, C, H, W = X.size()

    # Load a pre-trained ResNet model
    resnet = models.resnet18(pretrained=True)
    resnet = pretrain(model, args)
    # Remove the last layer of the ResNet model
    resnet = torch.nn.Sequential(*(list(resnet.children())[:-1])).eval()

    with torch.no_grad():
        S = torch.zeros((args.num_classes*m, C, H, W), dtype=X.dtype, device=args.device)
        for c in range(args.num_classes):
            # Extract features from the dataset using the ResNet model
            X_features = resnet(X)

            # Compute the empirical mean of the dataset
            mu = X_features.mean(dim=0)

            # Initialize the set of selected points and the set of unselected points
            U = X.clone()

            # Iteratively select the coreset points
            for i in range(m):
                # Extract features from the unselected points using the ResNet model
                U_features = resnet(U)

                # Compute the similarity between the unselected points and the empirical mean based on the features
                sim = F.cosine_similarity(U_features, mu.view(1, -1), dim=1)

                # Find the index of the unselected point with the highest similarity
                j = torch.argmax(sim)

                # Add the selected point to the coreset and remove it from the set of unselected points
                S[(m*c)+i] = U[j]
                U = torch.cat((U[:j], U[j+1:]))

                # Update the empirical mean based on the selected points
                mu = S[:(m*c)+i+1].mean(dim=0)
                mu_features = resnet(mu.unsqueeze(0)).squeeze(0)
                mu = mu_features

    return S