import numpy as np
import matplotlib.pyplot as plt
from math import pow
import datetime
# parameters
rad = 10
thk = 5
sep = -5


def prediction_value(d):
    g = np.sign(np.dot(np.transpose(w), d[:3]))
    if not ((g > 0 and d[3] > 0) or (g < 0 and d[3] < 0)):
        return False
    else:
        return True


def prediction_training(w):
    update = 1
    e_in = []
    updated_w = w
    while update <= 100000:
        is_classified = True
        for d in data:
            is_prediction = prediction_value(w, d)
            if not is_prediction:
                w += d[3] * d[:3]
                is_classified = False

                break
        update += 1
        e_in.append(pocket_input_error())
        if update > 2:
            if e_in[-1] < e_in[-2]:
                updated_w = w
            else:
                w = updated_w
                e_in[-1]=e_in[-2]
        if is_classified:
            break

    print("No of updates = " + str(update-1))
    print("\n weights =  ", w)
    return e_in


def pocket_input_error():
    no_of_error = 0
    for d in data:
        is_prediction = prediction_value(w, d)
        if not is_prediction:
            no_of_error += 1
    return no_of_error/len(data)


def linear_regression_input_error():
    total_error = 0
    for d in data:
        total_error += pow((np.dot(np.transpose(w), d[:3]) - d[-1]),2)

    return total_error/len(data)


def linear_regression_training():
    X_Input = []
    Y_predict = []
    for d in data:
        X_Input.append(d[:3])
        Y_predict.append(d[3])
    X_inverse = np.linalg.inv(np.dot(np.transpose(X_Input), X_Input))
    pseudo_X = np.dot(X_inverse,np.transpose(X_Input))
    w = np.dot(pseudo_X, Y_predict)
    return w



def graph_plot():
    plt.gcf().clear()
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


# n data points,(x1,y1) are the coordinates of the top semi-circle
def generatedata(rad, thk, sep, n, x1=0, y1=0):
    # center of the top semi-circle
    X1 = x1
    Y1 = y1

    # center of the bottom semi-circle
    X2 = X1 + rad + thk / 2
    Y2 = Y1 - sep

    # data points in the top semi-circle
    top = []
    # data points in the bottom semi-circle
    bottom = []

    # parameters
    r1 = rad + thk
    r2 = rad

    cnt = 1
    top_count = 1
    bottom_count = 1
    while (cnt <= n):
        # uniformed generated points
        x = np.random.uniform(-r1, r1)
        y = np.random.uniform(-r1, r1)

        d = x ** 2 + y ** 2
        if (d >= r2 ** 2) and (d <= r1 ** 2):
            if y > 0 :
                top.append([X1 + x, Y1 + y])
                cnt += 1
                top_count += 1

            else:
                #if bottom_count <= (n/2):
                    bottom.append([X2 + x, Y2 + y])
                    cnt += 1
                    bottom_count += 1

        else:
            continue

    return top, bottom


def execute_pocket():
    error = prediction_training(w)
    iterations = [i for i in range(100000)]
    plt.plot(iterations, error, 'r--')
    plt.show()
    graph_plot()

def polynomial():
    for d in data:
        zeeta.append([1, d[1], d[2], d[1]**2, d[1]*d[2], d[2]**2, d[1]**3, (d[1]**2)*d[2], (d[2]**2)*d[1]], d[2]**3)

    return zeeta

top, bottom = generatedata(rad, thk, sep, 2000)

X1 = [i[0] for i in top]
Y1 = [i[1] for i in top]

X2 = [i[0] for i in bottom]
Y2 = [i[1] for i in bottom]

# pre-processing the data for (a)
x1 = [[1] + i + [1] for i in top]
x2 = [[1] + i + [-1] for i in bottom]
data = x1 + x2

zeeta = []
data = np.array(data)
np.random.shuffle(data)
w = np.zeros(len(data[0]) - 1)
present_time = datetime.datetime.now()
execute_pocket()
final_time = datetime.datetime.now()
#graph_plot()
print("time taken for pla ")

print(final_time - present_time)
w = linear_regression_training()
present_time = datetime.datetime.now()
print("time taken for linear regression ")

print(present_time - final_time)
print(w)
print("error in linear regression = ")
print(linear_regression_input_error())
graph_plot()

polynomial()
