import numpy as np
import os
from keras.datasets import cifar10

import matplotlib.pyplot as plt

# 激活函数及其导数
def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu_derivative(z):
    return (z > 0).astype(float)

def sigmoid_derivative(a):
    return a * (1 - a)

def tanh_derivative(a):
    return 1 - a**2

class ThreeLayerNet:
    def __init__(self, input_size, hidden_size, output_size, activation='relu'):
        self.params = {}
        self.activation = activation
        
        # 初始化权重
        if activation == 'relu':
            init_factor = np.sqrt(2.0 / input_size)
        else:
            init_factor = np.sqrt(1.0 / input_size)
            
        self.params['W1'] = init_factor * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = np.sqrt(1.0 / hidden_size) * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def forward(self, X):
        self.z1 = np.dot(X, self.params['W1']) + self.params['b1']
        
        if self.activation == 'relu':
            self.a1 = relu(self.z1)
        elif self.activation == 'sigmoid':
            self.a1 = sigmoid(self.z1)
        elif self.activation == 'tanh':
            self.a1 = tanh(self.z1)
            
        self.z2 = np.dot(self.a1, self.params['W2']) + self.params['b2']
        exp_scores = np.exp(self.z2 - np.max(self.z2, axis=1, keepdims=True))
        self.probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return self.probs

    def backward(self, X, y, reg_lambda):
        num_samples = X.shape[0]
        
        delta3 = self.probs
        delta3[range(num_samples), y] -= 1
        delta3 /= num_samples

        dW2 = np.dot(self.a1.T, delta3)
        db2 = np.sum(delta3, axis=0)
        
        delta2 = np.dot(delta3, self.params['W2'].T)
        if self.activation == 'relu':
            delta2 *= relu_derivative(self.z1)
        elif self.activation == 'sigmoid':
            delta2 *= sigmoid_derivative(self.a1)
        elif self.activation == 'tanh':
            delta2 *= tanh_derivative(self.a1)

        dW1 = np.dot(X.T, delta2)
        db1 = np.sum(delta2, axis=0)

        # 添加L2正则化项
        dW2 += reg_lambda * self.params['W2']
        dW1 += reg_lambda * self.params['W1']

        return {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2}

    def compute_loss(self, X, y, reg_lambda):
        probs = self.forward(X)
        corect_logprobs = -np.log(probs[range(len(X)), y])
        data_loss = np.sum(corect_logprobs)
        data_loss += 0.5 * reg_lambda * (np.sum(np.square(self.params['W1'])) + 
                     np.sum(np.square(self.params['W2'])))
        return 1./len(X) * data_loss

def load_cifar10():
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    # 预处理
    X_train = X_train.astype('float32').reshape(-1, 32*32*3) / 255.0
    X_test = X_test.astype('float32').reshape(-1, 32*32*3) / 255.0
    y_train = y_train.flatten()
    y_test = y_test.flatten()
    
    # 划分验证集
    val_size = 5000
    X_val = X_train[:val_size]
    y_val = y_train[:val_size]
    X_train = X_train[val_size:]
    y_train = y_train[val_size:]
    
    return X_train, y_train, X_val, y_val, X_test, y_test




def train(model, X_train, y_train, X_val, y_val, 
          learning_rate=1e-3, reg_lambda=1e-3, 
          epochs=100, batch_size=64, lr_decay=0.95):
    
    best_val_acc = 0.0
    num_train = X_train.shape[0]
    iterations_per_epoch = max(num_train // batch_size, 1)
    
   
    train_loss_history = []
    val_loss_history = []
    val_acc_history = []
    
    for epoch in range(epochs):
        lr = learning_rate * (lr_decay ** epoch)
        indices = np.random.permutation(num_train)
        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]
        
        
        epoch_train_loss = 0
        
        for i in range(iterations_per_epoch):
            start = i * batch_size
            end = start + batch_size
            X_batch = X_shuffled[start:end]
            y_batch = y_shuffled[start:end]
            
            
            model.forward(X_batch)
            grads = model.backward(X_batch, y_batch, reg_lambda)
            
            
            for param in model.params:
                model.params[param] -= lr * grads[param]
            
           
            batch_loss = model.compute_loss(X_batch, y_batch, reg_lambda)
            epoch_train_loss += batch_loss

       
        avg_train_loss = epoch_train_loss / iterations_per_epoch
        train_loss_history.append(avg_train_loss)
        
        
        val_loss = model.compute_loss(X_val, y_val, reg_lambda)
        val_acc = compute_accuracy(model, X_val, y_val)
        val_loss_history.append(val_loss)
        val_acc_history.append(val_acc)
        
       
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            np.savez('best_model.npz', **model.params)
            print(f"New best val acc: {val_acc:.4f}, saving model...")
        print(f"Epoch {epoch+1}/{epochs} | lr: {lr:.5f} | "
              f"train loss: {avg_train_loss:.4f} | "
              f"val loss: {val_loss:.4f} | val acc: {val_acc:.4f}")
    
 
    plot_training_curves(train_loss_history, val_loss_history, val_acc_history)
    
    return train_loss_history, val_loss_history, val_acc_history  

def plot_training_curves(train_loss, val_loss, val_acc):
    """ 绘制训练过程曲线 """
    plt.figure(figsize=(12, 4))
    
    # 绘制loss曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(val_acc, label='Validation Accuracy', color='darkorange')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_curves.png') 
    plt.show()



def compute_accuracy(model, X, y):
    probs = model.forward(X)
    preds = np.argmax(probs, axis=1)
    return np.mean(preds == y)

def hyperparameter_search():
    hidden_sizes = [256, 512]
    learning_rates = [1e-3, 5e-3]
    reg_lambdas = [1e-4, 1e-3]
    
    results = []
    X_train, y_train, X_val, y_val, _, _ = load_cifar10()
    
    for hs in hidden_sizes:
        for lr in learning_rates:
            for reg in reg_lambdas:
                print(f"\nTraining with hs={hs}, lr={lr}, reg={reg}")
                model = ThreeLayerNet(3072, hs, 10)
                train(model, X_train, y_train, X_val, y_val,
                      learning_rate=lr, reg_lambda=reg, epochs=30)
                val_acc = compute_accuracy(model, X_val, y_val)
                results.append({
                    'hidden_size': hs,
                    'learning_rate': lr,
                    'reg_lambda': reg,
                    'val_acc': val_acc
                })
    
 
    best = max(results, key=lambda x: x['val_acc'])
    print("\nBest hyperparameters:")
    print(best)

def test():
    X_train, y_train, X_val, y_val, X_test, y_test = load_cifar10()
    model = ThreeLayerNet(3072, 512, 10)  
    model.params = np.load('best_model.npz')
    test_acc = compute_accuracy(model, X_test, y_test)
    print(f"Test Accuracy: {test_acc:.4f}")

if __name__ == '__main__':
    X_train, y_train, X_val, y_val, _, _ = load_cifar10()
    
    model = ThreeLayerNet(3072, 512, 10)
    
    
    train_loss, val_loss, val_acc = train(model, 
                                        X_train, y_train,
                                        X_val, y_val,
                                        epochs=100,
                                        learning_rate=1e-3)
    
    
    test()
   