import os
import cv2

# Define the data directory
DATA_DIR = './data'
os.makedirs(DATA_DIR, exist_ok=True)  # Ensure the base directory exists

number_of_classes = 50
dataset_size = 100

cap = cv2.VideoCapture(0)

for j in range(number_of_classes):   
    class_dir = os.path.join(DATA_DIR, str(j))

    # ✅ Skip processing if the folder already exists
    if os.path.exists(class_dir) and len(os.listdir(class_dir)) > 0:
        print(f'Skipping class {j}, folder already exists and contains data.')
        continue  # Move to the next class without modifying existing data

    # Create folder only if it doesn’t exist
    os.makedirs(class_dir, exist_ok=True)
    print(f'Collecting data for class {j}')

    # Wait for user confirmation before starting
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read from camera")
            break

        cv2.putText(frame, 'Ready? Press "Q" or Close Window!', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                    1.0, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('frame', frame)

        key = cv2.waitKey(1) & 0xFF  # Read key press
        if key == ord('q') or cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) < 1:
            cap.release()
            cv2.destroyAllWindows()
            exit()  # Ensure script exits

    # Capture and save dataset images
    for counter in range(dataset_size):
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow('frame', frame)

        # Detect if window is manually closed
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) < 1:
            cap.release()
            cv2.destroyAllWindows()
            exit()

        cv2.imwrite(os.path.join(class_dir, f'{counter}.jpg'), frame)

cap.release()
cv2.destroyAllWindows()


# # import os

# # import cv2


# # DATA_DIR = './data'
# # if not os.path.exists(DATA_DIR):
# #     os.makedirs(DATA_DIR)

# # number_of_classes = 50
# # dataset_size = 100

# # cap = cv2.VideoCapture(0)

# # for j in range(number_of_classes):   
# #     if not os.path.exists(os.path.join(DATA_DIR, str(j))):
# #         os.makedirs(os.path.join(DATA_DIR, str(j)))

# #     print('Collecting data for class {}'.format(j))

# #     done = False
# #     while True:
# #         ret, frame = cap.read()
# #         cv2.putText(frame, 'Ready? Press "Q" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
# #                     cv2.LINE_AA)
# #         cv2.imshow('frame', frame)
# #         if cv2.waitKey(25) == ord('q'):
# #             break

# #     counter = 0
# #     while counter < dataset_size:
# #         ret, frame = cap.read()
# #         cv2.imshow('frame', frame)
# #         cv2.waitKey(25)
# #         cv2.imwrite(os.path.join(DATA_DIR, str(j), '{}.jpg'.format(counter)), frame)

# #         counter += 1

# # cap.release()
# # cv2.destroyAllWindows()




# import os
# import cv2

# # Define the data directory
# DATA_DIR = './data'
# os.makedirs(DATA_DIR, exist_ok=True)  # Ensure the base directory exists

# number_of_classes = 50
# dataset_size = 100

# cap = cv2.VideoCapture(0)

# for j in range(number_of_classes):   
#     class_dir = os.path.join(DATA_DIR, str(j))

#     # ✅ Skip processing if the folder already exists
#     if os.path.exists(class_dir) and len(os.listdir(class_dir)) > 0:
#         print(f'Skipping class {j}, Folder already exists and contains data.')
#         continue  # Move to the next class without modifying existing data

#     # Create folder only if it doesn’t exist
#     os.makedirs(class_dir, exist_ok=True)
#     print(f'Collecting data for class {j}')

#     # Wait for user confirmation before starting
#     while True:
#         ret, frame = cap.read()
#         cv2.putText(frame, 'Ready? Press "Q" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 
#                     1.3, (0, 255, 0), 3, cv2.LINE_AA)
#         cv2.imshow('frame', frame)
#         if cv2.waitKey(25) == ord('q'):
#             break

#     # Capture and save dataset images
#     for counter in range(dataset_size):
#         ret, frame = cap.read()
#         cv2.imshow('frame', frame)
#         cv2.waitKey(25)
#         cv2.imwrite(os.path.join(class_dir, f'{counter}.jpg'), frame)

# cap.release()
# cv2.destroyAllWindows()

