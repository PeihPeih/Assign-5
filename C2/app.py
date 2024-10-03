from fastapi import FastAPI, Body
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import pickle
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import tensorflow as tf
import pandas as pd

app = FastAPI()

origins = ['*']
app.add_middleware(CORSMiddleware, allow_origins =origins,allow_credentials=True, allow_methods=['*'], allow_headers=["*"])

@app.get("/")
async def root():
    return "Hello world"


@app.post("/api/score/predict")
async def predict_weight(score1: int = Body(...), score2: int = Body(...), score3: int = Body(...),model: str = Body(...)):
    scaler = MinMaxScaler()
    df = pd.read_csv('./sinh_vien_data.csv')
    x = df[['Điểm 1', 'Điểm 2', 'Điểm 3']]
    scaler.fit(x)
    score_scaler = scaler.transform([[score1, score2, score3]])
    score1, score2, score3 = score_scaler[0]
    
    exam_score = 0
    if model == "cnn":
        class CNNModel(tf.Module):
            def __init__(self):
                super(CNNModel, self).__init__()
                self.conv1 = tf.Variable(tf.random.normal([1, 1, 1, 32]), name='conv1_weights', dtype=tf.float32)  # Lớp Conv2D với 32 filters
                self.dense1 = tf.Variable(tf.random.normal([96, 48]), name='dense1_weights', dtype=tf.float32)  # Fully connected layer 1
                self.dense2 = tf.Variable(tf.random.normal([48, 1]), name='dense2_weights', dtype=tf.float32)   # Fully connected layer 2

            def __call__(self, x):
                x = tf.reshape(x, [-1, 3, 1, 1])  # Reshape input thành [batch_size, height=2, width=1, channels=1]
                conv = tf.nn.conv2d(x, self.conv1, strides=[1, 1, 1, 1], padding="VALID")
                conv = tf.nn.relu(conv)  # Activation ReLU
                conv_flat = tf.reshape(conv, [-1, 96])  # Flatten
                dense1_out = tf.matmul(conv_flat, self.dense1)
                dense1_out = tf.nn.relu(dense1_out)  # Activation ReLU
                output = tf.matmul(dense1_out, self.dense2)  # Output layer
                return output

        model = CNNModel()
        conv1_weights_np = np.load('./cnn/conv1_weights.npy').astype(np.float32)
        dense1_weights_np = np.load('./cnn/dense1_weights.npy').astype(np.float32)
        dense2_weights_np = np.load('./cnn/dense2_weights.npy').astype(np.float32)

        model.conv1.assign(conv1_weights_np)    
        model.dense1.assign(dense1_weights_np)
        model.dense2.assign(dense2_weights_np)

        exam_score = model(np.array([score1, score2, score3], dtype=np.float32)).numpy()[0][0]
    elif model == "rnn":
        class SimpleRNN:
            def __init__(self, input_size, hidden_size):
                self.hidden_size = hidden_size
                self.Wxh = tf.Variable(tf.random.normal([input_size, hidden_size]), name='Wxh')
                self.Whh = tf.Variable(tf.random.normal([hidden_size, hidden_size]), name='Whh')
                self.bh = tf.Variable(tf.zeros([hidden_size]), name='bh')
                self.Why = tf.Variable(tf.random.normal([hidden_size, 1]), name='Why')
                self.by = tf.Variable(tf.zeros([1]), name='by')

            def step(self, x, h):
                print(x.shape, h.shape, self.Wxh.shape, self.Whh.shape)
                print(x.shape, h.shape, tf.matmul(x, self.Wxh), tf.matmul(h, self.Whh), self.bh)
                h = tf.tanh(tf.matmul(x, self.Wxh) + tf.matmul(h, self.Whh) + self.bh)
                y = tf.matmul(h, self.Why) + self.by
                return y, h

            def forward(self, x):
                h = tf.zeros([x.shape[0], self.hidden_size])  # Initialize hidden state
                # print(h.shape)
                for t in range(x.shape[1]):  # Loop over time steps
                    y, h = self.step(x[:, t, :], h)
                return y

            @property
            def trainable_variables(self):
                return [self.Wxh, self.Whh, self.bh, self.Why, self.by]
            
        model = SimpleRNN(3, 64)
        model.Wxh.assign(np.load('./rnn/Wxh.npy'))
        model.Whh.assign(np.load('./rnn/Whh.npy'))
        model.bh.assign(np.load('./rnn/bh.npy'))
        model.Why.assign(np.load('./rnn/Why.npy'))
        model.by.assign(np.load('./rnn/by.npy'))

        exam_score = model.forward(np.array([score1, score2, score3], dtype=np.float32).reshape(-1,1,3)).numpy()[0][0]
    elif model == "cnn-keras":
        model = tf.keras.models.load_model('./cnn.h5')
        exam_score = model.predict(np.array([[score1, score2, score3]], dtype=np.float32))[0][0]
    elif model == "rnn-keras":
        model = tf.keras.models.load_model('./rnn.h5')
        exam_score = model.predict(np.array([[score1, score2, score3]], dtype=np.float32))[0][0]
    else:
        with open("./model.pkl", "rb") as f:
            model = pickle.load(f)
        exam_score = model.predict([[score1, score2, score3]])[0]
    return {
        "exam_score": "{:.2f}".format(exam_score)
    }