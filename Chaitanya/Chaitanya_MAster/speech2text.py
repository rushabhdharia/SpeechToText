import tensorflow as tf

from data_generator import DataGenerator
from train_test_utils import model_evaluate, model_fit
from model import ASRModel

print(tf.__version__)
print(tf.test.is_gpu_available())

#Paths
train_data_path = "./LibriSpeech100/TRAIN_DATA.pkl"
train_label_path = "./LibriSpeech100/TRAIN_LABEL.pkl"
val_data_path = "./LibriSpeech100/DEV_DATA.pkl"
val_label_path = "./LibriSpeech100/DEV_LABEL.pkl"
test_data_path = "./LibriSpeech100/TEST_DATA.pkl"
test_label_path = "./LibriSpeech100/TEST_LABEL.pkl"







file = open(train_data_path, "rb")
TRAIN_DATA = pickle.load(file)
file = open(train_label_path, "rb")
TRAIN_LABELS = pickle.load(file)


file = open(val_data_path, "rb")
VAL_DATA = pickle.load(file)
file = open(val_label_path, "rb")
VAL_LABELS = pickle.load(file)




file = open(test_data_path,  "rb")
TEST_DATA = pickle.load(file)
file = open(test_label_path, "rb")
TEST_LABELS = pickle.load(file)



# Create DataGenerator Objects

train_data = DataGenerator(TRAIN_DATA,TRAIN_LABELS)
val_data = DataGenerator(VAL_DATA,VAL_LABELS)
test_data = DataGenerator(TEST_DATA,TEST_LABELS)


# Build Model
model = ASRModel()
# model.build(input_shape = [None, None, 20])
optimizer = tf.keras.optimizers.Adam()

# print(model.summary())

# Checkpoint
ckpt_dir = './training_checkpoints'
ckpt = tf.train.Checkpoint(epoch = tf.Variable(0), optimizer=optimizer, model = model)
manager = tf.train.CheckpointManager(ckpt, ckpt_dir, max_to_keep = 2)    

# Train Model
losses, accuracies, val_losses, val_acc = model_fit(model, optimizer, train_data, manager, ckpt, val_ds = val_data, epochs = 100)

# To Do - Add Plots

save_file = open('outputs/predictions.txt', 'w')
ckpt.restore(manager.latest_checkpoint)

_, acc = model_evaluate(model, test_data, test=True, save_file = save_file)

print("Training Accuracy = ", acc)

# To Do - Add Predict function