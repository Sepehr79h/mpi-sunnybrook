import torch


def load_data():
    # temporary sample data

    batch_size = 2

    train_x = torch.rand(200, 40, 64, 64)  # num images, num channels, img height, img width
    train_x = torch.unsqueeze(train_x, dim=1)
    test_x = torch.rand(10, 40, 64, 64)

    train_y = torch.randint(0, 4, (200,))
    test_y = torch.randint(0, 4, (10,))

    train_dataset = torch.utils.data.TensorDataset(train_x, train_y)
    test_dataset = torch.utils.data.TensorDataset(test_x, test_y)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
