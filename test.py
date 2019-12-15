import tensorflow as tf
from model import ASRModel
from train_test_utils import model_evaluate
from data_generator import DataGenerator

test_path = "./LibriSpeech100/test/test_all/"
test_data = DataGenerator(test_path)

model = ASRModel()
optimizer = tf.keras.optimizers.Adam()

ckpt_dir = './training_checkpoints'
ckpt = tf.train.Checkpoint(epoch=tf.Variable(1), optimizer=optimizer, model = model)
manager = tf.train.CheckpointManager(ckpt, ckpt_dir, max_to_keep = 2)

ckpt.restore(manager.latest_checkpoint)

save_file = open('outputs/predictions.txt', 'w')
_, acc = model_evaluate(model, test_data, test=True, save_file = save_file)    

print(acc)