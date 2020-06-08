from utils import load, get_labels
import numpy as np

def mostprobableclass(arr, classcounter, classindex, segments):
    """

    :param arr: list
        labels
    :param classcounter: list
        number of elements per class
    :param classindex: list
        indexes to jump to every class
    :param segments: int
        number of segments to evaluate every loop
    :return: list
        most likely class every x segments (majority voting)
    """
    t = 0
    ans = []
    for i, class_val in enumerate(classcounter):
        s = segments
        for h in range(class_val//s):
            s = t + segments
            hash_func = dict()
            for data_chunk in range(t, (s-1)):
                if arr[data_chunk] in hash_func.keys():
                    hash_func[arr[data_chunk]] += 1
                else:
                    hash_func[arr[data_chunk]] = 1
            max_count = 0
            ret = -1
            for g in hash_func:
                if max_count < hash_func[g]:
                    ret = g
                    max_count = hash_func[g]

            ans.append(ret)
            t += segments
        t = classindex[i] + 1
    return ans


real_labels = get_labels(load("../../pickles_modified/trainPickle4"))
predicted_labels = []
with open('predictions_best.txt', 'r') as predictions:
    for line in predictions:
        predicted_labels.append(int(line))

classes = ['Backhoe_JD50DCompact', 'Compactor_Ingersoll_Rand', 'Concrete_Mixer_0',
           'Excavator_CAT320E', 'Excavator_Hitachi50U']
predicted_labels = np.array(predicted_labels)
real_labels = np.array(real_labels)
class_counter = [np.count_nonzero(predicted_labels == i) for i in range(len(classes))]

major_counter = [np.count_nonzero(real_labels == i) for i in range(len(classes))]  # how many elements per class
major_index = []
ind = 0
for elem in major_counter:  # creating indexes to jump from class to class directly
    ind += elem
    major_index.append(ind)

sample = 17  # ----------------------------------------------------------------------------Majority voting amount!
major_pred_labels = mostprobableclass(predicted_labels, major_counter, major_index, sample)
major_real_labels = mostprobableclass(real_labels, major_counter, major_index, sample)

major_pred_labels = np.array(major_pred_labels)
major_real_labels = np.array(major_real_labels)

True_positive = [0] * 6  # TP for each class and sum
False_positive = [0] * 6  # FP for each class and sum
False_negative = [0] * 6  # FN for each class and sum
matrix = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]

raw = False  # if True, no majority voting is used

if raw:
    sample = 1
    for index in range(len(predicted_labels)):
        if predicted_labels[index] == real_labels[index]:
            True_positive[len(classes)] += 1
            True_positive[predicted_labels[index]] += 1
        else:
            False_positive[len(classes)] += 1
            False_positive[predicted_labels[index]] += 1

            False_negative[real_labels[index]] += 1
        matrix[real_labels[index]][predicted_labels[index]] += 1
else:
    class_counter = [np.count_nonzero(major_pred_labels == i) for i in range(len(classes))]
    for index in range(len(major_pred_labels)):
        if major_pred_labels[index] == major_real_labels[index]:
            True_positive[len(classes)] += 1
            True_positive[major_pred_labels[index]] += 1
        else:
            False_positive[len(classes)] += 1
            False_positive[major_pred_labels[index]] += 1

            False_negative[major_real_labels[index]] += 1
        matrix[major_real_labels[index]][major_pred_labels[index]] += 1

precision = [0] * len(classes)
recall = [0] * len(classes)
f1_score = [0] * len(classes)
average_precision = 0
average_recall = 0
average_f1_score = 0

for ind in range(len(classes)):
    precision[ind] = True_positive[ind] / (True_positive[ind] + False_positive[ind])
    recall[ind] = True_positive[ind] / (True_positive[ind] + False_negative[ind])
    f1_score[ind] = 2 * precision[ind] * recall[ind] / (precision[ind] + recall[ind])

for ind in range(len(classes)):
    average_precision += precision[ind]
    average_recall += recall[ind]
    average_f1_score += f1_score[ind]

average_precision = average_precision/len(classes) * 100
average_recall = average_recall/len(classes) * 100
average_f1_score = average_f1_score/len(classes)

real_number = 30000//sample

print("Total accuracy is: {} %  {}/{}\n".format(True_positive[len(classes)] / (real_number * len(classes)) * 100,
      True_positive[len(classes)], real_number * len(classes)))
for ind in range(len(classes)):
    print("Precision of class {} is: {} %   {}/{}".format(classes[ind], precision[ind] * 100,
                                                          True_positive[ind], class_counter[ind]))
print("\n")
for ind in range(len(classes)):
    print("Recall of class {} is: {} %   {}/{}".format(classes[ind], recall[ind] * 100,
          True_positive[ind], True_positive[ind] + False_negative[ind]))
print("\n")
for ind in range(len(classes)):
    print("F1-score of class {} is: {}".format(classes[ind], f1_score[ind]))
print("\n")
print("Average Precision is: {} %".format(average_precision))
print("Average Recall is: {} %".format(average_recall))
print("Average F1-score is: {}".format(average_f1_score))

for line in matrix:
    print(line)
print("\n")

for i, line in enumerate(matrix):
    for j, elem in enumerate(line):
        matrix[i][j] = round((matrix[i][j]/real_number)*100, 2)

for line in matrix:
    print(line)
