import math
import timeit
import matplotlib.pyplot as plt

def get_predictions_and_times(model,test_dataset):
    """
    Returns an array with the first element being the predicted y values and the second element being the time taken per prediction.
    """
    timer_start=timeit.default_timer()
    y_pred = model.predict(test_dataset)
    timer_end=timeit.default_timer()

    # check if predictions need to have first element unpacked (for instance, when BERT is being used to make predictions)
    if len(y_pred)==1:
        num_predictions = len(y_pred[0])
    else: 
        num_predictions = len(y_pred)
        
    time_per_prediction = (timer_end-timer_start)/num_predictions
    return y_pred, time_per_prediction

def round_to_seconds(x):
    """
    Round the input x to the first 3 significant digits and display small fractions with an exponent
    in the format 0.345e-4.
    """
    if x == 0:
        return 0
    else:
        exp = int(math.floor(math.log10(abs(x))))
        if exp >= -2:
            return round(x, -exp + 2)
        else:
            sigfigs = round(x * (10 ** (2 - exp)))
            return "{:.3f}e-{:02d}".format(sigfigs / 1000, -exp)
        

def print_accuracy_f1_time(accuracy_score,f1_score,time_per_prediction):
    """
    Prints the accuracy score, f1 score and time per prediction.
    """

    print(f"Accuracy: {accuracy_score:.2f}")
    print(f"F1 score: {f1_score:.2f}")
    print(f"Time per prediction: {round_to_seconds(time_per_prediction)} seconds")

def print_html_table(accuracy_NB_tfidf, bert_accuracy, accuracy_nn, f1_NB_tfidf, bert_f1_score, f1_nn, time_per_prediction_NB, time_per_prediction_BERT, time_per_prediction_NN):
    """
    Prints an html table comparing accuracy scores, f1 scores and the time taken per prediction for the respective classification models.
    """

    html_table= """
    ||  Naive Bayes Classifier | BERT Classifier| LSTM Neural Network|
    |----------|----------|----------|----------|
    Accuracy| {}  | {}  |{}|
    F1 Score| {}  | {}  |{}|
    Time per prediction| {} seconds | {} seconds |{} seconds|
    
    """.format(round(accuracy_NB_tfidf , 2), 
            round(bert_accuracy, 2), 
            round(accuracy_nn, 2) ,
            round(f1_NB_tfidf, 2), 
            round( bert_f1_score, 2),
            round(f1_nn, 2),
            round_to_seconds(time_per_prediction_NB),
            round_to_seconds(time_per_prediction_BERT),
            round_to_seconds(time_per_prediction_NN)
            )

    # print the HTML table
    return html_table

def plot_accuracy_loss(model_history,model_title):
    """
    Returns two line plots, the first showing Training and Validation Accuracy, while the second shows Training and Validation Loss.
    """
    # Extract the accuracy and loss values from the history object
    acc = model_history.history['accuracy']
    val_acc = model_history.history['val_accuracy']
    loss = model_history.history['loss']
    val_loss = model_history.history['val_loss']

    # Plot the accuracy and loss values
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='upper left')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()), 1])
    plt.title('Training and Validation Accuracy - '+model_title)

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper left')
    plt.ylabel('Cross Entropy')
    plt.ylim([0, 1.0])
    plt.title('Training and Validation Loss - '+model_title)
    plt.xlabel('epoch')
    plt.show()
