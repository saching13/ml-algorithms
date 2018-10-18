import random
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['axes.unicode_minus'] = False


def prediction_value(weights, d):
    g = np.dot(np.transpose(weights), d[:3])
    if not ((g > 0 and d[3] > 0) or (g < 0 and d[3] < 0)):
        return False
    else:
        return True


def training(weights):
    update = 0
    while True:
        is_classified = True
        for d in data:
            is_prediction = prediction_value(weights, d)
            if not is_prediction:
                weights += d[3] * d[:3]
                update += 1
                is_classified = False
            if update > 500:
                break
        if update > 500 or is_classified:
            break

    print("No of updates = " + str(update))
    print("\n weights =  ", w)


def graph_plot():
    plt.plot(X1, Y1, 'bo', X2, Y2, 'g*', [-1, 1], [1, -1], 'm')
    axes = plt.gca()
    x_intercept = np.array(axes.get_xlim())
    y_intercept = (-w[0] / w[2]) + (-w[1] / w[2]) * x_intercept
    plt.plot(x_intercept, y_intercept, '--')
    plt.ylabel("Y axis")
    plt.xlabel("X axis")
    plt.text(1, 1, "class +1")
    plt.text(-1, -1, "class -1")
    plt.show()


if __name__ == '__main__':
    # generate a data set of 20.
    # for simplicity, 10 in the first quadrant, another 10 in the third quadrant
    X1 = []
    Y1 = []
    X2 = []
    Y2 = []

    for i in range(20):
        X1.append(random.uniform(0, 1))
        Y1.append(random.uniform(0, 1))
        X2.append(random.uniform(-1, 0))
        Y2.append(random.uniform(-1, 0))

    # label the data
    data1 = [np.array([1, X1[i], Y1[i], 1]) for i in range(20)]
    data2 = [np.array([1, X2[i], Y2[i], -1]) for i in range(20)]
    data = data1 + data2

    w = np.zeros(len(data[0]) - 1)
    training(w)
    graph_plot()
