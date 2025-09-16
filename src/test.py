from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("runs/test")
for i in range(10):
    writer.add_scalar("Test/Scalar", i * 2, i)
writer.close()