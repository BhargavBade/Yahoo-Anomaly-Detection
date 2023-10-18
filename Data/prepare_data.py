from ccbdl.data.utils.get_loader import get_loader

def prepare_data(data_config):
    loader = get_loader(data_config["dataset"])
    train_data, test_data, val_data = loader(**data_config).get_dataloader()
    return train_data, test_data, val_data