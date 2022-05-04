# William Collins
# Methods to finetune pretrained torchvision models on new datasets.

import os, random

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch import nn, optim

from sklearn.metrics import classification_report, confusion_matrix

import seaborn as sns
import matplotlib.pyplot as plt



def finetune_model(model, optimizer, tensor_path, model_path, lab2idx, device, batch_size=32, n_epochs=20):
    '''Train the model on the new dataset. Use cross validation accuracy to checkpoint the model at improvements.
    Training data is in a directory containing stored tensors of 2000 images each. Use two of these files as test
    and cross validation. Shuffle the order of the files each epoch. '''
    model.train()
    criterion = nn.CrossEntropyLoss()
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=10)
    train_losses = []
    cv_losses = []
    
    if not os.path.exists(model_path):
        os.mkdir(model_path)
        
    xs = list(sorted([fn for fn in os.listdir(tensor_path) if 'x_' in fn and not '_0.' in fn]))
    ys = list(sorted([fn for fn in os.listdir(tensor_path) if 'y_' in fn and not '_0.' in fn]))   
    
    x_test = torch.load('%s/%s' % (tensor_path, xs[0]))
    y_test = torch.load('%s/%s' % (tensor_path, ys[0]))

    x_cv = torch.load('%s/%s' % (tensor_path, xs[1]))
    y_cv = torch.load('%s/%s' % (tensor_path, ys[1]))

    xs=xs[2:]
    ys=ys[2:]
    
    best_acc = 0
    
    for epoch in range(n_epochs):
        print('Epoch: %d' % epoch)
        
        idx = list(range(len(xs)))
        random.shuffle(idx)
        xs = [xs[i] for i in idx]
        ys = [ys[i] for i in idx]
        
        ep_ttl = 0
        for i in range(len(xs)):
            x = torch.load('%s/%s' % (tensor_path, xs[i]))
            y = torch.load('%s/%s' % (tensor_path, ys[i]))
            
            for j in range(0, x.size(0), batch_size):
                ep_ttl += 1
                x_batch = x[j:j+batch_size].to(device)
                y_batch = y[j:j+batch_size].to(device)
                
                #reset gradient for this iteration
                optimizer.zero_grad()

                #run the data through the model
                output = model(x_batch)

                #get the negative log likelihood loss
                loss = criterion(output, y_batch)

                #calculate the gradients
                loss.backward()

                #update the parameters from the gradients
                optimizer.step()
                
                if ep_ttl%100==0:
                    print('Epoch: %d, File: %s, Batch: %d, Loss: %.6f' % (epoch, xs[i], j, loss.item()))
                    train_losses.append(loss.item())
            
            # test the model on cross validation and store if there's an improvement in accuracy
            print('Testing model...')
            acc, cv_loss = test(model, criterion, x_cv, y_cv)
            scheduler.step(cv_loss)
            cv_losses.append(cv_loss)
            print('Test accuracy %.6f, prev best acc: %.6f %s' % (acc, best_acc, '!! IMPROVED !!' if acc>best_acc else ''))
        
            if acc>best_acc:
                best_acc = acc
                no_improvement = 0
                print('Saving model...')
                torch.save(model.state_dict(), '%s/model.pt' % model_path)
                torch.save(optimizer.state_dict(), '%s/optimizer.pt' % model_path)
            else:
                no_improvement += 1

            #Stop early if there's no improvement for several epochs (about 4)
            if no_improvement >= 40:
                print('no improvement in several epochs, breaking')
                break
            
    #load the best model and test on test data, providing a classification report and confusion matrix
    model.load_state_dict(torch.load('%s/model.pt' % model_path))
    test_acc, _ = test(model, criterion, x_test, y_test, lab2idx, True)
    print('final test accuracy: %.6f' % test_acc)
    model.eval()
    
    return model, train_losses, cv_losses

            
def test(model, criterion, x_test, y_test, lab2idx=None, print_report=False):
    '''Test the model and return accuracy and loss. Option to print classification report and confusion matrix.'''
    #set model to eval mode to turn off dropout etc.
    model.eval()
    
    correct = 0
    loss = 0
    with torch.no_grad():
        #get the test output
        output = model(x_test)
        
        #calculate the test loss
        loss = criterion(output, y_test)
        
        #select the indices of the maximum output values/prediction
        _, y_pred = torch.max(output, 1)

        #compare them with the target digits and sum correct predictions
        correct = y_pred.eq(y_test).sum()
        
    #calculate the accuracy
    acc = correct / y_test.size()[0]
    
    print('Test accuracy %.6f, %d of %d' % (acc, correct, y_test.size()[0]))
    
    if print_report:
        #print the classification report
        idx2lab = {v:k for k,v in lab2idx.items()}
        class_labels = [idx2lab[i] for i in range(len(idx2lab))]
        
        print('\n\n')
        print(classification_report(y_test.tolist(), y_pred.tolist(), target_names=class_labels, digits=4))
        print('\n\n')
    
        #create and display a confusion matrix
        cm = confusion_matrix(y_test.tolist(), y_pred.tolist())
        fig, ax = plt.subplots(figsize=(12,10))
        f = sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_labels, yticklabels=class_labels, ax=ax)
    
    model.train()
    
    return acc, loss.item()
