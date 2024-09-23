from fastapi import FastAPI, Body
from sklearn.preprocessing import LabelEncoder
import pickle
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import tensorflow as tf

app = FastAPI()

origins = ['*']
app.add_middleware(CORSMiddleware, allow_origins =origins,allow_credentials=True, allow_methods=['*'], allow_headers=["*"])

@app.get("/")
async def root():
    return "Hello world"

@app.get("/api/jobs/get_all")
async def get_all_jobs():
    return ["doctor", "dentist", "engineer", "teacher", "lawyer", "nurse", "pharmacist", "architect", 
        "chef", "scientist", "musician", "artist", "pilot", "firefighter", "police officer", 
        "accountant", "software developer", "mechanic", "electrician", "plumber", 
        "salesperson", "marketing manager", "graphic designer", "journalist", 
        "data analyst", "psychologist", "biologist", "chemist", "physicist", "economist"]

@app.post("/api/weight/predict")
async def predict_weight(height: float = Body(...), job: str = Body(...), model: str = Body(...)):
    jobs = await get_all_jobs()
    lb = LabelEncoder()
    lb.fit(jobs)
    job_encode = lb.transform([job])[0]
    weight_predict = 0
    if model == "cnn":
        class CNNModel(tf.Module):
            def __init__(self):
                super(CNNModel, self).__init__()
                self.conv1 = tf.Variable(tf.random.normal([1, 1, 1, 32]), name='conv1_weights', dtype=tf.float32)  # Lớp Conv2D với 32 filters
                self.dense1 = tf.Variable(tf.random.normal([64, 64]), name='dense1_weights', dtype=tf.float32)  # Fully connected layer 1
                self.dense2 = tf.Variable(tf.random.normal([64, 1]), name='dense2_weights', dtype=tf.float32)   # Fully connected layer 2

            def __call__(self, x):
                x = tf.reshape(x, [-1, 2, 1, 1])  # Reshape input thành [batch_size, height=2, width=1, channels=1]
                conv = tf.nn.conv2d(x, self.conv1, strides=[1, 1, 1, 1], padding="VALID")
                conv = tf.nn.relu(conv)  # Activation ReLU
                conv_flat = tf.reshape(conv, [-1, 64])  # Flatten
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

        weight_predict = model(np.array([[height, job_encode]], dtype=np.float32)).numpy()[0][0]
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
                h = tf.tanh(tf.matmul(x, self.Wxh) + tf.matmul(h, self.Whh) + self.bh)
                y = tf.matmul(h, self.Why) + self.by
                return y, h

            def forward(self, x):
                h = tf.zeros([x.shape[0], self.hidden_size])  # Initialize hidden state
                for t in range(x.shape[1]):  # Loop over time steps
                    y, h = self.step(x[:, t, :], h)
                return y

            @property
            def trainable_variables(self):
                return [self.Wxh, self.Whh, self.bh, self.Why, self.by]
            
        model = SimpleRNN(2, 64)
        model.Wxh.assign(np.load('./rnn/Wxh.npy').astype(np.float32))
        model.Whh.assign(np.load('./rnn/Whh.npy').astype(np.float32))
        model.bh.assign(np.load('./rnn/bh.npy').astype(np.float32))
        model.Why.assign(np.load('./rnn/Why.npy').astype(np.float32))
        model.by.assign(np.load('./rnn/by.npy').astype(np.float32))

        weight_predict = model.forward(np.array([[height, job_encode]], dtype=np.float32).reshape(-1,1,2)).numpy()[0][0]
    else:
        with open("./model.pkl", "rb") as f:
            model = pickle.load(f)
        weight_predict = model.predict([[height, job_encode]])[0]
    bmi = weight_predict / (height ** 2)
    return {"weight": '{:.2f}'.format(weight_predict),
            "bmi": '{:.2f}'.format(bmi)}