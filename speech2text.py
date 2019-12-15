import tensorflow as tf

from data_generator import DataGenerator
from train_test_utils import model_evaluate, model_fit
from model import ASRModel

print(tf.__version__)
print(tf.test.is_gpu_available())

#Paths
train_path = "./LibriSpeech100/train/train_all/"
dev_path = "./LibriSpeech100/dev/dev_all/"
test_path = "./LibriSpeech100/test/test_all/"

# Create DataGenerator Objects
train_data = DataGenerator(train_path)
val_data = DataGenerator(dev_path)
test_data = DataGenerator(test_path)

# Build Model
model = ASRModel()
# model.build(input_shape = [None, None, 20])
optimizer = tf.keras.optimizers.Adam()

# print(model.summary())

# Checkpoint
ckpt_dir = './training_checkpoints'
ckpt = tf.train.Checkpoint(optimizer=optimizer, model = model)
manager = tf.train.CheckpointManager(ckpt, ckpt_dir, max_to_keep = 2)    

# Train Model
losses, accuracies, val_losses, val_acc = model_fit(model, optimizer, train_data, manager, ckpt, val_ds = val_data, epochs = 100)

# To Do - Add Plots

save_file = open('outputs/predictions.txt', 'w')
ckpt.restore(manager.latest_checkpoint)

_, acc = model_evaluate(model, test_data, test=True, save_file = save_file)

print("Training Accuracy = ", acc)

# To Do - Add Predict function