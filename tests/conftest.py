import torch

def pytest_sessionstart(session):
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
