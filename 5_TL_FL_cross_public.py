import pandas as pd
import optuna
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Input, Dropout, Reshape, LSTM
from tensorflow.keras.layers import LayerNormalization
from tensorflow.keras.optimizers import Adam
from keras.utils import plot_model
from matplotlib import pyplot
from numpy.random import seed
seed(3407)
tf.random.set_seed(3407)
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, mean_absolute_error, r2_score
import time
from sklearn.preprocessing import MinMaxScaler
import os
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
import sys
from tensorflow.keras.models import load_model, Model
import matplotlib.pyplot as plt
import scienceplots

   
def_font_size = 18
def_label_size = 16
def_legend_size = 12

def_n_trials = 20
def_epoch = 800
test_size_1 = 0.1       
Output_path = r'Output\test_cnn\\'


def read_data(path, split):

    data = pd.read_excel(path, sheet_name="Sheet2")
    data =data.values
    scaler = MinMaxScaler(feature_range=(0, 1))
    XY = scaler.fit_transform(data)
    X_train, X_test, y_train, y_test = train_test_split(XY[:, 1:], XY[:, 0],      
                                                        test_size=split, random_state=3407, shuffle=False)

    X_train = X_train.reshape(X_train.shape[0],-1,1)
    X_test = X_test.reshape(X_test.shape[0],-1,1)
    y_train = y_train.reshape(y_train.shape[0],-1,1)
    y_test = y_test.reshape(y_test.shape[0],-1,1)
    X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
    y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
    X_test = tf.convert_to_tensor(X_test, dtype=tf.float32)
    y_test = tf.convert_to_tensor(y_test, dtype=tf.float32)

    return X_train, X_test, y_train, y_test

## In this process, Bayesian optimization is used to select the hyperparameters during model building,
## and the hyperparameters of the fine-tuned model retain the previously obtained optimized parameters.
def objective(trial):
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)      
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])        
    num_nodes = trial.suggest_categorical('num_nodes',[16,32,64])

    inputs = Input(shape=(X_train.shape[1],X_train.shape[2]))
    cnn1 = Conv1D(filters=num_nodes, kernel_size=1, activation='relu')(inputs)  
    cnn1 = MaxPooling1D(pool_size=2)(cnn1)  
    cnn2 = Conv1D(filters=num_nodes, kernel_size=1, activation='relu')(cnn1)  
    cnn2 = MaxPooling1D(pool_size=2)(cnn2)  
    cnn3 = Conv1D(filters=num_nodes, kernel_size=1, activation='relu')(cnn2)  
    cnn4 = Conv1D(filters=num_nodes, kernel_size=1, activation='relu')(cnn3)  
    cnn = Flatten()(cnn4)  
    dense1 = Dense(X_train.shape[1] * X_train.shape[2], activation='relu')(cnn)  
    output = Dense(1, activation='linear')(dense1)

    model = Model(inputs=inputs, outputs=output)
    model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=learning_rate))
    history = model.fit(X_train, y_train, epochs=def_epoch, batch_size=batch_size, 
                        validation_data=(X_test, y_test), verbose=0, shuffle=False) # 

# Evaluation Criteria for Iterative Model Optimization
    val_losses = history.history['val_loss']
    for epoch, loss in enumerate(val_losses):  
            trial.report(loss, step=epoch)
            # Early stop
            if trial.should_prune():
                raise optuna.TrialPruned()
    return val_losses[-1]


mean_values_list = []
std_values_list = []
rmse_values_list = []
cell_list = ['CS33', 'CS34', 'CS35', 'CS36', 'CS37', 'CS38', 'CX34', 'CX36', 'CX37' ]

for index1, value_cell in enumerate(cell_list):
    print(f"{index1+1} Round, Load model {value_cell}")
    input_file = r'HI\CALCE_HI\HI_{}_new.xlsx'.format(value_cell)
    X_train, X_test, y_train, y_test = read_data(input_file, test_size_1)
    sys.stdout = open(Output_path + 'output.txt','w')
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=def_n_trials)     

    records = []
    for trial in study.trials:
        record = {
            'Trial_Number': trial.number,
            'Value': trial.value,
            'State': trial.state,
        }
        record.update(trial.params)
        records.append(record)

    df_trials = pd.DataFrame(records)

    intermediate_dict = {}
    for trial in study.trials:
        trial_name = f'Trial_{trial.number}'
        intermediate_dict[trial_name] = trial.intermediate_values

    df_intermediate = pd.DataFrame.from_dict(intermediate_dict, orient='index')
    df_intermediate = df_intermediate.transpose()  
    df_intermediate.reset_index(inplace=True)
    df_intermediate.rename(columns={'index': 'Step'}, inplace=True)

    os.makedirs(Output_path, exist_ok=True)
    with pd.ExcelWriter(os.path.join(Output_path, "{}_Bayes_optuna_plot.xlsx".format(value_cell))) as writer:
        df_trials.to_excel(writer, sheet_name="Trial_Params", index=False)
        df_intermediate.to_excel(writer, sheet_name="Intermediate_Values", index=False)


    best_params = study.best_params
    best_learning_rate = best_params['learning_rate']
    best_batch_size = best_params['batch_size']
    best_num_nodes = best_params['num_nodes']

    df = pd.DataFrame({
        'best_learning_rate': [best_learning_rate],
        'best_batch_size': [best_batch_size],
        'best_num_nodes': [best_num_nodes]
    })
    df.to_excel(Output_path +f'{value_cell}_20Beyes_CNN_best_params.xlsx', index=False)

    best_learning_rates = {}
    best_batch_sizes = {}
    best_learning_rates[value_cell] = best_learning_rate
    best_batch_sizes[value_cell] = best_batch_size


    inputs = Input(shape=(X_train.shape[1], X_train.shape[2]))
    cnn1 = Conv1D(filters=best_num_nodes, kernel_size=1, activation='relu')(inputs)  
    cnn1 = MaxPooling1D(pool_size=2)(cnn1)  
    cnn2 = Conv1D(filters=best_num_nodes, kernel_size=1, activation='relu')(cnn1) 
    cnn2 = MaxPooling1D(pool_size=2)(cnn2) 
    cnn3 = Conv1D(filters=best_num_nodes, kernel_size=1, activation='relu')(cnn2) 
    cnn4 = Conv1D(filters=best_num_nodes, kernel_size=1, activation='relu')(cnn3)  
    cnn = Flatten()(cnn4)  
    dense1 = Dense(X_train.shape[1] * X_train.shape[2], activation='relu')(cnn)  
    output = Dense(1, activation='linear')(dense1)

    best_model = Model(inputs=inputs, outputs=output)
    best_model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=best_learning_rate))
    best_model.summary()
    best_model.save(Output_path + "100trained_{}_optim20_CNN.h5".format(value_cell))

    ## Model information
    trainable_count = np.sum([np.prod(v.get_shape()) for v in best_model.trainable_weights])
    non_trainable_count = np.sum([np.prod(v.get_shape()) for v in best_model.non_trainable_weights])
    input_shape = X_train.shape[1:]  
    full_model = tf.function(lambda x: best_model(x))
    concrete_func = full_model.get_concrete_function(tf.TensorSpec([1] + list(input_shape), tf.float32))
    frozen_func = convert_variables_to_constants_v2(concrete_func)
    graph_def = frozen_func.graph.as_graph_def()
    with tf.compat.v1.Graph().as_default() as graph:
        tf.graph_util.import_graph_def(graph_def, name='')
        flops = tf.compat.v1.profiler.profile(
            graph=graph,
            options=tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        )
    Forward_flops = flops.total_float_ops
    Train_flops = flops.total_float_ops * 2
    total_count = trainable_count + non_trainable_count

    # Train
    start_time = time.time()       
    history = best_model.fit(X_train, y_train, epochs=def_epoch, batch_size=best_batch_size, 
                        validation_data=(X_test, y_test), verbose=0, shuffle=False) 
    end_time = time.time()            
    training_time = end_time - start_time

    # Prediction
    predictions = best_model.predict(X_test)
    X_train = X_train.numpy().flatten()
    y_train = y_train.numpy().flatten()
    y_test = y_test.numpy().flatten()
    predictions_save = predictions.flatten() 

    # Save Predictions Data
    df = pd.DataFrame({
        'test_actual': y_test,
        'test_predicted': predictions_save
    })
    df.to_excel(Output_path + 'Actual VS Predicted.xlsx', index=False)

    # Evaluation
    mse = mean_squared_error(y_test, predictions)
    mape = mean_absolute_percentage_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)
    ae = np.abs(y_test - predictions)
    df_ae = pd.DataFrame(ae)
    df_ae.to_excel(Output_path + 'ae.xlsx', index=False) 
    sd = np.std(ae)
    aae = np.mean(ae)

    df = pd.DataFrame()
    df['metrics'] = ['MSE', 'MAPE', 'MAE', 'RMSE', 'R^2', 'Absolute Error (AE)', 
                'Standard Deviation of AE (SD)', 'Average Absolute Error (AAE)', 'Training Time (seconds)',
                'Total_count', 'Trainable_count', 'Non_trainable_count', 'Forward FLOPs', 'Train FLOPs*']# 
    df['value'] = [mse, mape, mae, rmse, r2, ae, sd, aae, training_time, 
                    total_count, trainable_count, non_trainable_count, Forward_flops, Train_flops]  #
    df.to_excel(Output_path + f'{value_cell}_Beyesian_Model_evaluation.xlsx', index=False)

    sys.stdout.close()
    sys.stdout = sys.__stdout__



    # best_model_parameters
    best_learning_rate = best_learning_rates[value_cell]
    best_batch_size = best_batch_sizes[value_cell]

    train_list = [10, 20, 30, 40]
    layer_list = [1,2,3,4,5,6,7,8,9]  

    value_cell_list = [cell for cell in cell_list if cell != value_cell]
    for value_cell_2 in value_cell_list:
        print(f"{index1+1} Round, Test cell {value_cell_2}")
        for index2, value_train in enumerate(train_list):
            for index3, value_layer in enumerate(layer_list):
                ## Plot style
                plt.style.use(['science','nature','no-latex'])      
                plt.rcParams['font.family'] = 'Arial'  
                plt.rcParams['axes.unicode_minus'] = False  
                fig, axes = plt.subplots(3, 1, figsize=(4, 8)) 

                new_input_file = r'HI\NASA_HI\HI_{}_new.xlsx'.format(value_cell_2)
                Output_path_2 = f'Output\\test_cnn\{value_cell}-{value_cell_2}_cnn\\train_split_{value_train}\layer{value_layer}\\'
                if not os.path.exists(Output_path_2):
                    os.makedirs(Output_path_2)
                # Output to file
                sys.stdout = open(Output_path_2 + 'output.txt','w')
                
                X_train_n, X_test_n, y_train_n, y_test_n = read_data(new_input_file, 1-value_train/100)

                new_model = load_model(r"Output\test_cnn\100trained_{}_optim20_CNN.h5".format(value_cell))     
                ## Load model
                for layer in new_model.layers[:value_layer]:
                    layer.trainable = False

                new_model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=best_learning_rate))
                new_model.summary()

                ## Model information
                trainable_count = np.sum([np.prod(v.get_shape()) for v in new_model.trainable_weights])
                non_trainable_count = np.sum([np.prod(v.get_shape()) for v in new_model.non_trainable_weights])
                input_shape = X_train_n.shape[1:]  
                full_model = tf.function(lambda x: new_model(x))
                concrete_func = full_model.get_concrete_function(tf.TensorSpec([1] + list(input_shape), tf.float32))
                frozen_func = convert_variables_to_constants_v2(concrete_func)
                graph_def = frozen_func.graph.as_graph_def()
                with tf.compat.v1.Graph().as_default() as graph:
                    tf.graph_util.import_graph_def(graph_def, name='')
                    flops = tf.compat.v1.profiler.profile(
                        graph=graph,
                        options=tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
                    )
                Forward_flops = flops.total_float_ops
                Train_flops = flops.total_float_ops * 2
                total_count = trainable_count + non_trainable_count

                # Train
                start_time = time.time()       
                history = new_model.fit(X_train_n, y_train_n, epochs=def_epoch, batch_size=best_batch_size, 
                                    validation_data=(X_test_n, y_test_n), verbose=0, shuffle=False) # 
                end_time = time.time()           
                training_time = end_time - start_time

                # Prediction
                predictions = new_model.predict(X_test_n)
                predictions = predictions.flatten() 
                        
                X_train = X_train_n.numpy().flatten()
                y_train = y_train_n.numpy().flatten()
                y_test = y_test_n.numpy().flatten()

                # Evaluation
                mse = mean_squared_error(y_test, predictions)
                mape = mean_absolute_percentage_error(y_test, predictions)
                mae = mean_absolute_error(y_test, predictions)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test, predictions)
                ae = np.abs(y_test - predictions)
                df_ae = pd.DataFrame(ae)
                df_ae.to_excel(Output_path_2 + 'ae.xlsx', index=False) # ae
                sd = np.std(ae)
                aae = np.mean(ae)
          
                print('Best batch size:', best_batch_size)
                print('Best learn rate:', best_learning_rate)

                df = pd.DataFrame()
                df['metrics'] = ['MSE', 'MAPE', 'MAE', 'RMSE', 'R^2', 'Absolute Error (AE)', 
                            'Standard Deviation of AE (SD)', 'Average Absolute Error (AAE)', 'Training Time (seconds)',
                            'Total_count', 'Trainable_count', 'Non_trainable_count', 'Forward FLOPs', 'Train FLOPs*']# 
                df['value'] = [mse, mape, mae, rmse, r2, ae, sd, aae, training_time, 
                            total_count, trainable_count, non_trainable_count, Forward_flops, Train_flops]  #
                df.to_excel(Output_path_2 + 'Model_evaluation.xlsx', index=False)


                ## 14 Model Loss 
                ax = axes[0]
                ax.plot(history.history['loss'], label='train', linestyle='--',linewidth=2)
                ax.plot(history.history['val_loss'], label='val', linewidth=2)
                ax.set_xlabel('Epoch',fontsize=def_font_size)
                ax.set_ylabel('Loss',fontsize=def_font_size)
                # Adjusting Axis Label Font Size
                ax.tick_params(axis='x', labelsize=def_label_size)
                ax.tick_params(axis='y', labelsize=def_label_size)
                ax.legend(fontsize=def_legend_size)

                ## Predictions VS Actual
                ax = axes[1]
                alpha_values = 1-abs(predictions-y_test.reshape(-1))  
                alpha_values = np.clip(alpha_values, 0, 1)
                ax.scatter(y_test, predictions, color='blue', edgecolor='k', s=50, alpha=alpha_values, label='Result')
                ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', linewidth=2, label='X=Y')
                ax.set_xlabel('Actual Capacity',fontsize=def_font_size)
                ax.set_ylabel('Predicted Capacity',fontsize=def_font_size)
                # Adjusting Axis Label Font Size
                ax.tick_params(axis='x', labelsize=def_label_size)
                ax.tick_params(axis='y', labelsize=def_label_size)
                ax.legend(fontsize=def_legend_size)

                ## Error interval
                ax = axes[2]
                train_steps = range(len(y_train))   
                test_steps = range(len(y_train), len(y_train) + len(y_test))   
                train_real_values = y_train   
                test_real_values = y_test   
                train_predicted_values = new_model.predict(X_train_n).flatten()  
                test_predicted_values = predictions
                train_error = np.abs(train_real_values - train_predicted_values) / train_real_values * 100   
                test_error = np.abs(test_real_values - test_predicted_values) / test_real_values * 100  

                # Error interval 5%
                error_threshold = 5  
                ax.plot(train_steps, train_real_values, label="Train Real Capacity", linewidth=2)
                ax.plot(train_steps, train_predicted_values, label="Train Predicted Capacity", linewidth=2)
                ax.scatter(test_steps, test_real_values, label="Test Real Capacity",s=10)
                ax.scatter(test_steps, test_predicted_values, label="Test Predicted Capacity",s=10)
                ax.fill_between(train_steps, train_real_values * (1 - error_threshold/100), 
                                    train_real_values * (1 + error_threshold/100), alpha=0.2, color='blue', label='Train Error')
                ax.fill_between(test_steps, test_real_values * (1 - error_threshold/100), 
                                    test_real_values * (1 + error_threshold/100), alpha=0.2, color='orange', label='Test Error')

                # Error interval 3%
                error_threshold = 3  
                ax.fill_between(train_steps, train_real_values * (1 - error_threshold/100), 
                                    train_real_values * (1 + error_threshold/100), alpha=0.1, color='blue', edgecolor='black')
                ax.fill_between(test_steps, test_real_values * (1 - error_threshold/100), 
                                    test_real_values * (1 + error_threshold/100), alpha=0.1, color='orange', edgecolor='black')
                
                ax.set_xlabel("Time Steps",fontsize=def_font_size)
                ax.set_ylabel("Capacity",fontsize=def_font_size)
                ax.tick_params(axis='x', labelsize=def_label_size)
                ax.tick_params(axis='y', labelsize=def_label_size)
                ax.legend(fontsize=def_legend_size)
                print('Predicted results have been plotted')

                # Save Predictions Data
                df = pd.DataFrame({
                    'test_steps':test_steps,
                    'test_actual': test_real_values,
                    'test_predicted': test_predicted_values
                })
                df.to_excel(Output_path_2 + 'Actual VS Predicted.xlsx', index=False)

                plt.tight_layout()  
                plt.savefig(Output_path_2 + 'Model_prediction.png', dpi=300)
                print('The results of the model transfer fine-tuning have been output')

                sys.stdout.close()
                sys.stdout = sys.__stdout__
