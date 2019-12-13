import numpy as np
import tensorflow as tf
from tensorflow.keras.backend import ctc_batch_cost, ctc_decode, get_value
from utils import wer, indices_to_string


def validate(model, x, y_true, input_len, label_len, y_strings, test = False):
    input_len = np.expand_dims(input_len, axis = 1)
    label_len = np.expand_dims(label_len, axis = 1)
    
    y_pred = model(x)
    loss = ctc_batch_cost(y_true, y_pred, input_len, label_len)
    
    input_len = np.squeeze(input_len)
    y_decode = ctc_decode(y_pred, input_len)[0][0]
    
    accuracy = 0.0
    
    for i in range(len(y_strings)):
        predicted_sentence = indices_to_string(get_value(y_decode[i]).numpy())
        accuracy += wer(predicted_sentence, y_strings[i])
        
        if test:
            print("Correct Sentence:", y_strings[i])
            print("Predicted Sentence:", predicted_sentence)
    
    return tf.reduce_mean(loss), accuracy/len(y_strings)    


def model_evaluate(model, val_ds, test = False):
    val_step = 0
    val_loss = 0.0
    val_accuracy = 0.0
            
    for inputs, y in val_ds:
        x, y_strings, ip_len, label_len = inputs
        val_step += 1       
        loss, accuracy = validate(model, x, y, ip_len, label_len, y_strings, test)
        val_loss += loss
        val_accuracy += accuracy
                
    val_loss /= val_step
    val_accuracy /= val_step

    tf.print(' Validation Loss:', val_loss, ' Validation WER: ', val_accuracy)
    
    return val_loss, val_accuracy


def train_one_step(model, optimizer, x, y_true, input_len, label_len, y_strings):
#     print('------------------------------')
#     print(x.shape)
#     print(y.shape)
#     print(input_len.shape)
#     print(label_len.shape)
    
    input_len = np.expand_dims(input_len, axis = 1)
    label_len = np.expand_dims(label_len, axis = 1)
#     print(input_len.shape)
#     print(label_len.shape)
            
    with tf.GradientTape() as tape:
        y_pred = model(x)
#         print(y_pred.shape)
        loss = ctc_batch_cost(y_true, y_pred, input_len, label_len)
    
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    
    input_len = np.squeeze(input_len)
    y_decode = ctc_decode(y_pred, input_len)[0][0]
    
#         print(y_decode)
#         print(len(y_strings))
    
    accuracy = 0.0
    
    for i in range(len(y_strings)):
        predicted_sentence = indices_to_string(y_decode[i].numpy())
#             print(predicted_sentence)
        accuracy += wer(predicted_sentence, y_strings[i])
            
    return tf.reduce_mean(loss), accuracy/len(y_strings)


def model_fit(model, optimizer, train_ds, manager, val_ds = None,epochs=20):
    
    losses = []
    accuracies = []
    val_losses = []
    val_acc = []
    
    
    for epoch in range(epochs):
        step = 0
        epoch_loss = 0.0
        epoch_accuracy = 0.0
        for inputs, y in train_ds:
            x, y_strings, ip_len, label_len = inputs
            step += 1
            loss, accuracy = train_one_step(model, optimizer, x, y, ip_len, label_len, y_strings)
            epoch_loss += loss
            epoch_accuracy += accuracy
            if step % 78 == 0:
                print(step)
                
            
        epoch_loss /= step
        epoch_accuracy /= step
        
        losses.append(epoch_loss)
        accuracies.append(epoch_accuracy)
        
        tf.print('Epoch: ', epoch+1, ' Loss:', epoch_loss, ' WER: ', epoch_accuracy)
        
        
        if val_ds:
            val_loss, val_accuracy = model_evaluate(model, val_ds)
            val_losses.append(val_loss)
            val_acc.append(val_accuracy)
            
        if (epoch + 1) % 2 == 0:
            manager.save()
        
                
    if not val_ds:    
        return losses, accuracies
    
    return losses, accuracies, val_losses, val_acc