import cv2
import numpy
import numpy as np
import time
import matplotlib.pyplot as plt

def fill_gaps(arr, max_gap=50):
    result = arr.copy()
    start_index = -1
    end_index = -1

    for i, val in enumerate(result):
        if val == 1:
            if start_index == -1:
                start_index = i
            else:
                end_index = i
                result[start_index:end_index + 1] = 1
                start_index = -1
        elif start_index != -1 and i - start_index > max_gap:
            start_index = -1

    return result

def processVideo(cap, line_left, line_right, noline_left, noline_right):
    sens = 75
    min = 100
    max = 1000
    ret, prev_frame = cap.read()
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.GaussianBlur(prev_gray, (5, 5), 0)  # Добавляем фильтр Гаусса
    # Создаем пустой массив для хранения результатов обнаружения движения видео
    motion_left = []
    motion_right = []
    no_motion_left = []
    no_motion_right = []
    # Создаем пустой массив для хранения времени видео
    time_stamps = []
    start_time = time.time()
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        delta = cv2.absdiff(prev_gray, gray)

        # Применяем пороговую фильтрацию
        thresh = cv2.threshold(delta, sens, 255, cv2.THRESH_BINARY)[1]

        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.erode(thresh, kernel, iterations=1)
        thresh = cv2.dilate(thresh, kernel, iterations=1)
        left_roi = thresh[left_y1:left_y1 + left_h1, left_x1:left_x1 + left_w1]
        right_roi = thresh[right_y1:right_y1 + right_h1, right_x1:right_x1 + right_w1]
        # Ищем контуры
        contours_left, _ = cv2.findContours(left_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_right, _ = cv2.findContours(right_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Обнаружено ли движение?
        if len(contours_left) > 0:
            # Отфильтровываем контуры по площади
            filtered_contours_left = [cnt for cnt in contours_left if min <= cv2.contourArea(cnt) <= max]
            if len(filtered_contours_left) > 0:
                motion_left.append(line_left)
                no_motion_left.append(np.nan)
            else:
                motion_left.append(np.nan)
                no_motion_left.append(noline_left)
        else:
            motion_left.append(np.nan)
            no_motion_left.append(noline_left)

        # Обнаружено ли движение?
        if len(contours_right) > 0:
            # Отфильтровываем контуры по площади
            filtered_contours_right = [cnt for cnt in contours_right if min <= cv2.contourArea(cnt) <= max]
            if len(filtered_contours_right) > 0:
                motion_right.append(line_right)
                no_motion_right.append(np.nan)
            else:
                motion_right.append(np.nan)
                no_motion_right.append(noline_right)
        else:
            motion_right.append(np.nan)
            no_motion_right.append(noline_right)

        # Добавляем текущее время в массив time_stamps
        current_time = time.time() - start_time
        time_stamps.append(current_time)

        # Обновляем предыдущий кадр и его оттенки серого
        prev_frame = frame
        prev_gray = gray

        cv2.imshow('left', left_roi)
        cv2.imshow('right', right_roi)

        # Завершение цикла, если нажата клавиша 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    motion_left = np.array(motion_left)
    motion_right = np.array(motion_right)
    motion_left = fill_gaps(motion_left, 90)
    motion_right = fill_gaps(motion_right, 90)

    no_motion_left = np.array(no_motion_left)
    no_motion_right = np.array(no_motion_right)
    no_motion_left = fill_gaps(no_motion_left, 90)
    no_motion_right = fill_gaps(no_motion_right, 90)

    temp = time_stamps, motion_left, motion_right, no_motion_left, no_motion_right

    return temp

# Захватываем видео 1 и 2
cap1 = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(0)

#обрезка для 1 видео
left_x1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH) * 0.1)
left_y1 = 0
left_w1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH) * 0.2)
left_h1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))

right_x1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH) * 0.8)
right_y1 = 0
right_w1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH) * 0.2)
right_h1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))

#обрезка для второго видео
left_x2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH) * 0.1)
left_y2 = 0
left_w2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH) * 0.2)
left_h2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))

right_x2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH) * 0.8)
right_y2 = 0
right_w2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH) * 0.2)
right_h2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))

time_stamps1, motion_left1, motion_right1,no_motion_left1, no_motion_right1 = processVideo(cap1, 1, 4, 1, 4)
cap1.release()
cv2.destroyAllWindows()

time_stamps2, motion_left2, motion_right2, no_motion_left2, no_motion_right2 = processVideo(cap2, 3, 2, 3, 2)
cap2.release()
cv2.destroyAllWindows()

# Удаляем лишние элементы из motion_left1
motion_left1 = motion_left1[:len(time_stamps1)]
# Удаляем лишние элементы из motion_right1
motion_right1 = motion_right1[:len(time_stamps1)]

# Удаляем лишние элементы из no_motion_left1
no_motion_left1 = no_motion_left1[:len(time_stamps1)]

# Удаляем лишние элементы из no_motion_right1
no_motion_right1 = no_motion_right1[:len(time_stamps1)]

# Удаляем лишние элементы из motion_left2
motion_left2 = motion_left2[:len(time_stamps2)]

# Удаляем лишние элементы из motion_right2
motion_right2 = motion_right2[:len(time_stamps2)]

# Удаляем лишние элементы из no_motion_left2
no_motion_left2 = no_motion_left2[:len(time_stamps2)]

# Удаляем лишние элементы из no_motion_right2
no_motion_right2 = no_motion_right2[:len(time_stamps2)]

# Create new lists to store the masked "no motion" data
no_motion_masked_left1 = []
no_motion_masked_right1 = []
no_motion_masked_left2 = []
no_motion_masked_right2 = []

# Mask the "no motion" data where there is motion
for ml, mr, nml, nmr in zip(motion_left1, motion_right1, no_motion_left1, no_motion_right1):
    if not np.isnan(ml):
        nml = np.nan
    no_motion_masked_left1.append(nml)

    if not np.isnan(mr):
        nmr = np.nan
    no_motion_masked_right1.append(nmr)

for ml, mr, nml, nmr in zip(motion_left2, motion_right2, no_motion_left2, no_motion_right2):
    if not np.isnan(ml):
        nml = np.nan
    no_motion_masked_left2.append(nml)

    if not np.isnan(mr):
        nmr = np.nan
    no_motion_masked_right2.append(nmr)

# Plot masked "no motion" data
plt.figure()
plt.plot(time_stamps1, no_motion_masked_left1, linewidth=3)
plt.plot(time_stamps1, no_motion_masked_right1, linewidth=3)
plt.plot(time_stamps2, no_motion_masked_left2, linewidth=3)
plt.plot(time_stamps2, no_motion_masked_right2, linewidth=3)
plt.xlabel('Time (s)')
plt.ylabel('No Motion')
plt.title('No Motion Detection - Video')
plt.ylim(0, 5)
plt.xlim(0, 30)


# Строим график движения
plt.figure()
plt.plot(time_stamps1, motion_left1, linewidth=3)
plt.plot(time_stamps1, motion_right1, linewidth=3)
plt.plot(time_stamps2, motion_left2, linewidth=3)
plt.plot(time_stamps2, motion_right2, linewidth=3)
plt.xlabel('Time (s)')
plt.ylabel('Motion')
plt.title('Motion Detection - Video')
plt.ylim(0, 5)
plt.xlim(0, 30)

plt.show()