from concrete.ml.deployment import FHEModelClient, FHEModelDev, FHEModelServer
from concrete.ml.sklearn import XGBClassifier
from utils import clean_directory
import numpy
import time
import joblib

clean_directory("artifacts")

# Number of input to take into account for prediction
N_PREDICTIONS = 2

# Import preprocessed dataset
X = numpy.genfromtxt('dataset/X_training_preprocessed.csv', delimiter=',', skip_header=1)

# Number of samples to take into account for quantization setting
N_INPUT_SET = X.shape[0]
print(X.shape)

# Import clear model
clear_model = joblib.load("model/xgboost_model_classifier_binary.pkl")

start_time = time.time()
cml_model = XGBClassifier.from_sklearn_model(clear_model, X[:N_INPUT_SET, :], n_bits=14)
elapsed_time = (time.time() - start_time) / 60
print(f"Time elapsed for from_sklearn_model: {elapsed_time:.2f} minutes")

start_time = time.time()
cml_model.compile(X[:N_INPUT_SET, :])

elapsed_time = (time.time() - start_time) / 60
print(f"Time elapsed for compile: {elapsed_time:.2f} minutes")

# Save FHE model
fhemodel_dev = FHEModelDev('artifacts', cml_model)
fhemodel_dev.save()


# Let's create the client and load the model
fhemodel_client = FHEModelClient('artifacts', key_dir='keys')

# The client first need to create the private and evaluation keys.
serialized_evaluation_keys = fhemodel_client.get_serialized_evaluation_keys()
print(f"Evaluation keys size: {len(serialized_evaluation_keys) / (10**6):.2f} MB")


def client_send_input_to_server_for_prediction(encrypted_input):
        """Send the input to the server and execute on the server in FHE."""
        time_begin = time.time()
        encrypted_prediction = FHEModelServer('artifacts').run(
            encrypted_input, serialized_evaluation_keys
        )
        time_end = time.time()
        return time_end - time_begin, encrypted_prediction



# We create a loop to send the input to the server and receive the encrypted prediction
decrypted_predictions = []
execution_time = []
for i in range(N_PREDICTIONS):
    clear_input = X[[i], :]

    encrypted_input = fhemodel_client.quantize_encrypt_serialize(clear_input)

    exec_time, encrypted_prediction = client_send_input_to_server_for_prediction(encrypted_input)
    execution_time += [exec_time]

    decrypted_prediction = fhemodel_client.deserialize_decrypt_dequantize(encrypted_prediction)[0]

    decrypted_predictions.append(decrypted_prediction)

# Check MB size with sys of the encrypted data vs clear data
print(
    f"Encrypted data is "
    f"{len(encrypted_input)/clear_input.nbytes:.2f}"
    " times larger than the clear data"
)

# Show execution time
print(f"The average execution time is {numpy.mean(execution_time):.2f} seconds per sample.")


preds = clear_model.predict(X[:N_PREDICTIONS])

# Let's check the results and compare them against the clear model
clear_prediction_classes = cml_model.predict_proba(X[:N_PREDICTIONS]).argmax(axis=1)
decrypted_predictions_classes = numpy.array(decrypted_predictions).argmax(axis=1)

print("Concrete: FHE disable predictions")
print(clear_prediction_classes)

print("Concrete: FHE enable predictions")
print(decrypted_predictions_classes)

print("Original model predictions")
print(preds)