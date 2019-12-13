from data_generator import DataGenerator

train_path = "./LibriSpeech100/train/train_all/"
dev_path = "./LibriSpeech100/dev/dev_all/"
test_path = "./LibriSpeech100/test/test_all/"

train_data = DataGenerator(train_path)
val_data = DataGenerator(dev_path)
test_data = DataGenerator(test_path)

x, y = train_data[0]
x, y_strings, input_len, label_len = x

print("x", x.shape, x.dtype)
print("input_len", input_len.shape, input_len.dtype)
print("label_len", label_len.shape, label_len.dtype)
print("y", y.shape, y.dtype)
print("number of y_strings", len(y_strings))
print("y_string", y_strings[0])
