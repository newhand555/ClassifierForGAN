import torch
import torchvision
from torchvision import transforms, utils
from torch.utils.data import DataLoader
import torchvision.models as models
from torch.autograd import Variable
from tqdm import tqdm

def train():
    train_data = torchvision.datasets.ImageFolder(
        './dataset/horse2zebra/train',
        transform=transforms.Compose([
            transforms.Resize(286),
            transforms.RandomCrop((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    )
    print(len(train_data))
    val_data = torchvision.datasets.ImageFolder(
        './dataset/horse2zebra/val',
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    )
    print(len(val_data))
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_data, shuffle=True)


    model = models.resnet18(pretrained=True)
    fc_features = model.fc.in_features
    model.fc = torch.nn.Linear(fc_features, 2)

    model = model.cuda()

    for para in model.parameters():
        para.requires_grad = False
    # print(list(model.parameters())[-2:])
    # print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    # print(list(model.fc.parameters()))
    # print(model.parameters())
    # optimizer = torch.optim.SGD(params=[model.fc.weight, model.fc.bias], lr=1e-4, momentum=0.1)
    for para in model.fc.parameters():
        para.requires_grad = True
    optimizer = torch.optim.SGD(params=model.fc.parameters(), lr=1e-4, momentum=0.1)
    criterion = torch.nn.CrossEntropyLoss()

    # print(model)

    model.train()

    print(len(train_loader), len(val_loader))

    for epoch in range(100):
        total_loss = 0.0
        correct = 0
        for data in train_loader:
            inputs, labels = data
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs.data, 1)
            # print(preds, outputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.data

            correct += torch.sum(preds == labels.data).to(torch.float32)
        if ((epoch + 1) % 20 == 0):
            print('Epoch: {}, Loss: {}, Acc: {}'.format(epoch+1, total_loss, (correct/len(train_data))))

    correct = 0
    model.eval()
    for data in val_loader:
        inputs, labels = data
        # print(inputs.shape, labels.shape)
        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)
        # print(preds, labels)
        correct += torch.sum(preds == labels.data).to(torch.float32)
    print('Final Validation Acc: {}'.format((correct/len(val_data))))
    # print(correct, len(test_data))

    torch.save(model.state_dict(), './classification_dict_oa.pkl')
    torch.save(model, './classification_entire_oa.pkl')

def test():
    test_data = torchvision.datasets.ImageFolder(
        './dataset/apple2orange/test',
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    )
    print(len(test_data))
    test_loader = DataLoader(test_data, shuffle=True)
    model = models.resnet18(pretrained=True)
    fc_features = model.fc.in_features
    model.fc = torch.nn.Linear(fc_features, 2)
    model.load_state_dict(torch.load('./classification_dict_oa.pkl'))
    model = model.cuda()
    correct = 0
    model.eval()

    # print(test_data[0])

    for data in test_loader:
        inputs, labels = data
        # print(inputs.shape, labels.shape)
        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)
        # print(preds, labels)
        correct += torch.sum(preds == labels.data).to(torch.float32)
    print('Final Test Acc: {}'.format((correct / len(test_data))))












    # val_data = torchvision.datasets.ImageFolder(
    #     './dataset/horse2zebra/val',
    #     transform=transforms.Compose([
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                              std=[0.229, 0.224, 0.225])
    #     ])
    # )
    # print(len(val_data))
    # val_loader = DataLoader(val_data, shuffle=True)
    # correct = 0
    # for data in val_loader:
    #     inputs, labels = data
    #     # print(inputs.shape, labels.shape)
    #     inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
    #     outputs = model(inputs)
    #     _, preds = torch.max(outputs.data, 1)
    #     # print(preds, labels)
    #     correct += torch.sum(preds == labels.data).to(torch.float32)
    # print('Final Validation Acc: {}'.format((correct / len(val_data))))












def main():
    flag = False
    if flag == True:
        train()
    else:
        test()

    return

if __name__ == '__main__':
    main()