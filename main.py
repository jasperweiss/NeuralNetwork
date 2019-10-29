import numpy as np
import pickle
import csv
from PIL import Image
import matplotlib.pyplot as plt
import time


def save():
    with open('network_state.pkl', 'wb') as output:
        network_state = hiddenWeights, outputWeights
        pickle.dump(network_state, output)
        output.close()
        print('network state saved')


def load():
    with open('network_state.pkl', 'rb') as input:
        global hiddenWeights
        global outputWeights
        network_state = pickle.load(input)
        hiddenWeights = network_state[0]
        outputWeights = network_state[1]
        print('Network state loaded')


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoidPrime(x):
    return(sigmoid(1-sigmoid(x)))


# ----config----
inputNeurons = 784
outputNeurons = 10
learningRate = 0.001
verbose = False


def randomWeights(hiddenNeurons):
    global hiddenWeights
    global outputWeights
    hiddenWeights = 2 * np.random.random((hiddenNeurons, inputNeurons)) - 1
    outputWeights = 2 * np.random.random((outputNeurons, hiddenNeurons)) - 1


def predict(inputs):
    hiddenOutput = sigmoid(np.dot(hiddenWeights, inputs))
    output = sigmoid(np.dot(outputWeights, hiddenOutput))
    return(output)


def backpropagate(inputData, targetData):
    global outputWeights
    global hiddenWeights

    hiddenOutput = sigmoid(np.dot(hiddenWeights, inputData))
    output = sigmoid(np.dot(outputWeights, hiddenOutput))

    outputErrors = targetData - output
    hiddenErrors = np.dot(outputWeights.T, outputErrors)

    # Weight adjustments
    outputWeights = outputWeights + learningRate * \
        np.dot(outputErrors * sigmoidPrime(output), hiddenOutput.T)
    hiddenWeights = hiddenWeights + learningRate * \
        np.dot(hiddenErrors * sigmoidPrime(hiddenOutput), inputData.T)


def train(rounds):
    start = time.time()
    for y in range(0, rounds):
        with open('mnist_train.csv', newline='') as csvfile:
            data = csv.reader(csvfile)
            x = 0
            for record in data:
                if (x % 200 == 0) and (verbose is True):
                    progress = round((x/60000)*100, 2)
                    progressTotal = round(((y*60000+x)/(rounds*60000)*100), 2)
                    print("progress: "+str(progress)+"%, round("+str(y+1) +
                          "/"+str(rounds)+") ("+str(progressTotal)+"% total)")

                inputs = np.array([])
                targetData = np.array([])

                # Array with target valus for numbers 0-9
                targetData = np.zeros([10])
                targetData[int(record[0])] = float(1)

                # Array with input data
                record = np.delete(record, 0)
                for i in record:
                    converted = ((float(i) / float(255))*0.99+0.01)
                    inputs = np.append(inputs, converted)

                # Adding dimensions to arrays
                inputs = inputs.reshape((inputs.shape[0], 1))
                targetData = targetData.reshape((targetData.shape[0], 1))

                backpropagate(inputs, targetData)
                x = x + 1
            csvfile.close()
    end = time.time()
    minutes = round(((end - start)/60), 2)
    print("done training ("+str(minutes)+" minutes)")


def evaluate():
    start = time.time()
    with open('mnist_test.csv', newline='') as csvfile:
        data = csv.reader(csvfile)
        tested = 0
        correct = 0
        score = 0
        for record in data:
            inputs = np.array([])
            targetData = (float(record[0]))

            record = np.delete(record, 0)
            for i in record:
                converted = ((float(i) / float(255))*0.99+0.01)
                inputs = np.append(inputs, converted)
            inputs = inputs.reshape((inputs.shape[0], 1))

            predictions = predict(inputs)
            predictionsCertainty = np.array([])
            for x in predictions:
                value = x[0]
                predictionsCertainty = np.append(predictionsCertainty, x)
            guess = np.argmax(predictions)
            tested = tested + 1
            certainty = str((max(predictionsCertainty)*100))+"%"
            if (guess == targetData):
                correct = correct + 1
                score = (correct/tested*100)

        end = time.time()
        minutes = round(((end - start)/60), 2)
        print("done ("+str(minutes)+" minutes)")
        return(score)


def guess():
    load()
    imageFile = Image.open('writing.png').convert('LA')
    image = np.array(imageFile)
    plain = np.array([])
    for y in range(0, 28):
        for x in range(0, 28):
            plain = np.append(plain, image[y, x, 0])

    image_new = np.array(np.split(plain, 28))

    bitmap = Image.fromarray(image_new)

    inputs = np.array([])
    for c in plain:
        corrected = c/255
        inputs = np.append(inputs, corrected)

    inputs = inputs.reshape((inputs.shape[0], 1))
    predictions = predict(inputs)
    predictionsCertainty = np.array([])
    for x in predictions:
        value = x[0]
        predictionsCertainty = (np.append(predictionsCertainty, x))
    certainty = str(round(max(predictionsCertainty)*100, 2))+'%'
    guess = np.argmax(predictions)
    print("that's a "+str(guess)+' ('+certainty+' sure)')


def testProgress(hiddenNeurons, trainingRange):
    randomWeights(hiddenNeurons)
    scores = np.array([])
    x = np.array([])
    print("training network with "+str(hiddenNeurons) +
          " hidden neurons, "+str(trainingRange)+" rounds")
    print("starting 0 measurement..")
    score = evaluate()
    scores = np.append(scores, score)
    print("score: "+str(score))
    x = np.append(x, 0)
    start = time.time()
    for i in range(1, trainingRange+1):
        print("training configuration with "+str(hiddenNeurons) +
              " hidden neurons, round "+str(i)+"/"+str(trainingRange))
        train(1)
        print("evaluating accuracy for round "+str(i)+"/"+str(trainingRange))
        score = evaluate()
        scores = np.append(scores, score)
        x = np.append(x, i)
        print("score: "+str(score))
    results = (x, scores)
    end = time.time()
    minutes = round(((end - start)/60), 2)
    duration = minutes/trainingRange
    return(results, duration)


def analyze(configurations):
    trainingRange = 10
    print("testing configurations ")
    for configuration in configurations:
        start = time.time()
        results, duration = testProgress(configuration, trainingRange)
        data = (results, duration)

        filename = 'results_'+str(configuration)+'.pkl'

        with open(filename, 'wb') as output:
            pickle.dump(data, output)
            output.close()
        end = time.time()
        minutes = round(((end - start)/60), 2)
        print('training results for network with '+str(configuration) +
              ' hidden neuron(s) saved. ('+str(minutes)+' minutes total, '+str(duration)+' on average)')


def show(configurations):
    durations = np.array([])
    for configuration in configurations:
        filename = 'results_'+str(configuration)+'.pkl'
        with open(filename, 'rb') as output:
            data = pickle.load(output)
            results = data[0]
            duration = data[1]
            durations = np.append(durations, duration)
            print('configuration: '+str(configuration) +
                  " (average training+testing time: "+str(duration)+" minutes)")
            print(results)

        plt.plot(results[0], results[1])
        plt.xlabel('Training rounds --->')
        plt.ylabel('Accurcacy score (%) --->')
        plt.title('Training progression | ' +
                  str(configuration)+' hidden neurons')
        plt.show()

    plt.plot(configurations, durations)
    plt.xlabel('Hidden neurons --->')
    plt.ylabel('Duration per training in minutes --->')
    plt.title('Training duration')
    plt.show()
