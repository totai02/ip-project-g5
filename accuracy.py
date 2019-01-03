from src.extract_math_eq import *
import time
import cv2

# img = cv2.imread("data/Equations/Clean/eq3_hr.jpg")
# eq_string = extract_mat_eq(img, False)
path = "data/accuracy/"
labels = list()
durations = list()
scores = list()

# read label
for i in range(1, 14):
    label = []
    with open("data/clean_label/eq{}_hr.txt".format(i), 'r') as f:
        for line in f:
            line = line.replace("\n", "")
            if line == "":
                continue
            label.append(line)
    labels.append(np.array(label))

labels = np.array(labels)


# Origin Dataset
duration = list()
score = list()
print("Origin Dataset.\n")
for i in range(1, 14):
    img = cv2.imread(path + "origin/" + "eq{}_hr.jpg".format(i))

    start_time = time.time()
    eq_string, eq_chars = extract_mat_eq(img, False)
    delta_time = time.time() - start_time

    eq_chars = [char for char in eq_chars['char']]

    duration.append(delta_time)

    mean_score = sum(np.array(eq_chars) == labels[i - 1][:-1]) / len(eq_chars)

    score.append(mean_score)

    print(eq_string + " : " + "{:.4f} : {:.4f}s".format(mean_score, delta_time))

score = np.array(score)
duration = np.array(duration)

print("Origin Dataset Done.\n\n\n")


# Write to file
with open(path + "origin/" + "origin.txt", 'w') as f:
    f.write(str(np.sum(score) / len(score)))
    f.write(" ")
    f.write(str(np.sum(duration)/len(duration)) + "\n")
    for i in range(len(duration)):
        f.write("eq{}_hr.jpg".format(i + 1) + " " + str(score[i]) + " " + str(duration[i]) + "\n")


# Rotate Dataset
for angle in [-15, -10, -5, 5, 10, 15]:
    duration = list()
    score = list()
    print("Rotate {} Dataset.\n".format(angle))
    for i in range(1, 14):
        img = cv2.imread(path + "rotate{}/eq{}_hr.jpg".format(angle, i))

        start_time = time.time()
        eq_string, eq_chars = extract_mat_eq(img, False)
        delta_time = time.time() - start_time

        eq_chars = [char for char in eq_chars['char']]

        while len(eq_chars) < len(labels[i - 1][:-1]):
            eq_chars.append("")

        duration.append(delta_time)

        mean_score = sum(np.array(eq_chars) == labels[i - 1][:-1]) / len(eq_chars)

        score.append(mean_score)

        print(eq_string + " : " + "{:.4f} : {:.4f}s".format(mean_score, delta_time))

    score = np.array(score)
    duration = np.array(duration)

    print("Rotate {} Dataset Done.\n\n\n".format(angle))

    # Write to file
    with open(path + "rotate{}/rotate{}.txt".format(angle, angle), 'w') as f:
        f.write(str(np.sum(score) / len(score)))
        f.write(" ")
        f.write(str(np.sum(duration) / len(duration)) + "\n")
        for i in range(len(duration)):
            f.write("eq{}_hr.jpg".format(i + 1) + " " + str(score[i]) + " " + str(duration[i]) + "\n")


# Scale Dataset
for scale in [0.5, 2.0]:
    duration = list()
    score = list()
    print("Scale {} Dataset.\n".format(scale))
    for i in range(1, 14):
        img = cv2.imread(path + "scale{}/eq{}_hr.jpg".format(scale, i))

        start_time = time.time()
        eq_string, eq_chars = extract_mat_eq(img, False)
        delta_time = time.time() - start_time

        eq_chars = [char for char in eq_chars['char']]

        while len(eq_chars) < len(labels[i - 1][:-1]):
            eq_chars.append("")

        duration.append(delta_time)

        mean_score = sum(np.array(eq_chars) == labels[i - 1][:-1]) / len(eq_chars)

        score.append(mean_score)

        print(eq_string + " : " + "{:.4f} : {:.4f}s".format(mean_score, delta_time))

    score = np.array(score)
    duration = np.array(duration)

    print("Scale {} Dataset Done.\n\n\n".format(scale))

    # Write to file
    with open(path + "scale{}/scale{}.txt".format(scale, scale), 'w') as f:
        f.write(str(np.sum(score) / len(score)))
        f.write(" ")
        f.write(str(np.sum(duration) / len(duration)) + "\n")
        for i in range(len(duration)):
            f.write("eq{}_hr.jpg".format(i + 1) + " " + str(score[i]) + " " + str(duration[i]) + "\n")

