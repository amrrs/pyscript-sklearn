import joblib
import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, log_loss
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import normalize
from sklearn.utils import shuffle
from utils import summarize_training, visualize_training


# USER INPUTS
input_epochs = int(float(Element('input-epochs').element.value))
input_hidden_layer_sizes = Element('input-hidden-layer-sizes').element.value
input_hidden_layer_sizes_formatted = tuple(int(size) for size in input_hidden_layer_sizes.replace(' ', '').split(','))

print(f"Training for {input_epochs} epochs.")
print(f"Using {len(input_hidden_layer_sizes_formatted)} hidden layers with sizes {input_hidden_layer_sizes_formatted}.")

# LOADING DATA
data = load_iris(as_frame=False)
X, y, classes = normalize(data['data']), data['target'], list(range(0, len(np.unique(data['target']))))
int_labels_map = dict(zip(classes, data['target_names']))

# TRAINING PARAMETERS
HIDDEN_LAYER_SIZES = input_hidden_layer_sizes_formatted
EPOCHS = input_epochs
BATCH_SIZE = 8
LEARNING_RATE = 0.005
RANDOM_STATE = 42
MODEL_SAVE_PATH = 'model.pkl'
num_iter = int(len(X) / BATCH_SIZE)

# ALGORITHM
nn = MLPClassifier(
    batch_size=BATCH_SIZE,
    learning_rate_init=LEARNING_RATE,
    hidden_layer_sizes=HIDDEN_LAYER_SIZES,
    random_state=RANDOM_STATE
)

# TRAINING
accuracy_train = []
loss_train = []

print_epoch = lambda e: f"Running Epoch {e}..."

for e in range(0, EPOCHS):
    if EPOCHS < 10:
        print_epoch(e)
    else:
        # For larger epoch numbers, only print every tenth
        if e % 10 == 0:
            print_epoch(e)

    X, y = shuffle(X, y, random_state=e)
    batch_start = 0
    for i in range(0, num_iter):
        batch_end = batch_start + BATCH_SIZE
        X_batch, y_batch = X[batch_start:batch_end], y[batch_start:batch_end]
        nn.partial_fit(X_batch, y_batch, classes=classes)
        batch_start = batch_end

    y_pred = nn.predict(X)
    y_pred_prob = nn.predict_proba(X)

    accuracy_train.append(accuracy_score(y, y_pred))
    loss_train.append(log_loss(y, y_pred_prob, labels=classes))

print("-------------------------")
print("--TRAINING FINISHED--")
print("-------------------------")
_ = joblib.dump(nn, MODEL_SAVE_PATH)
summarize_training(EPOCHS, loss_train, accuracy_train, MODEL_SAVE_PATH)
visualize_training(loss_train, accuracy_train)
