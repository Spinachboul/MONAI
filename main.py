import monai
import torch
import torch.optim as optim
from monai.data import DataLoader, Dataset
from monai.networks.nets import UNet
from monai.transforms import Compose, LoadImage, AddChannel, ScaleIntensity, ToTensor
from monai.utils import first
from monai.losses import DiceLoss
from monai.metrics import DiceMetric as dice_metric

# Define transformers
transformers = Compose([
    LoadImage(image_only=True),
    AddChannel(),
    ScaleIntensity(),
])

# Define dataset and dataloader
images = ['image1.nii', 'image2.nii', 'image3.nii']
labels = ['label1.nii', 'label2.nii', 'label3.nii']
ds = Dataset(data=[(img, seg) for img, seg in zip(images, labels)], transform=transformers)
loader = DataLoader(ds, batch_size=2, shuffle=True, num_workers=4, pin_memory=torch.cuda.is_available())

# Create UNet, DiceLoss and Adam optimizer
device = torch.device("cuda:0")
model = UNet(
    dimensions=3,
    in_channels=1,
    out_channels=1,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
)
model.to(device)
loss_function = DiceLoss(sigmoid=True)
optimizer = optim.Adam(model.parameters(), 1e-3)

# start a typical PyTorch training
max_epochs = 6
val_interval = 2
best_metric = -1
best_metric_epoch = -1
epoch_loss_values = []
metric_values = []
post_pred = monai.transforms.Compose([monai.transforms.Activations(sigmoid=True)])
post_label = monai.transforms.Compose()

for epoch in range(max_epochs):
    print("-" * 10)
    print(f"epoch {epoch + 1}/{max_epochs}")
    model.train()
    epoch_loss = 0
    step = 0
    for batch_data in loader:
        step += 1
        inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        print(f"{step}/{len(loader)}, train_loss: {loss.item():.4f}")
    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

    if (epoch + 1) % val_interval == 0:
        model.eval()
        with torch.no_grad():
            metric_sum = 0.
            metric_count = 0
            for val_data in loader:
                val_inputs, val_labels = val_data[0].to(device), val_data[1].to(device)
                val_outputs = model(val_inputs)
                val_outputs = post_pred(val_outputs)
                val_labels = post_label(val_labels)
                value = dice_metric(y_pred=val_outputs, y=val_labels)
                metric_count += len(value)
                metric_sum += value.sum().item()
            metric = metric_sum / metric_count
            metric_values.append(metric)
            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1
                torch.save(model.state_dict(), "best_metric_model.pth")
                print("saved new best metric model")
            print(
                f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                f"\nbest mean dice: {best_metric:.4f} at epoch: {best_metric_epoch}"
            )
print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")