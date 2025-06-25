import pickle
import cv2
import mediapipe as mp
import numpy as np
import time

# Load the model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Start webcam
cap = cv2.VideoCapture(0)

# Set up MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.3)

# Label dictionary
labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 
               10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 
               19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 27: 'TIME NOW', 
               28: 'THANK YOU', 29: 'SORRY', 30: 'HELP', 31: 'MOM', 32: 'DAD', 33: 'I LOVE U', 
               34: 'PRAGATI', 35: 'MANVENDRA', 36: ''}

# Variables for tracking predictions
sentence = []
current_sign = ""
start_time = None
frame_skip = 3  # Predict every 3rd frame
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    frame_count += 1
    if frame_count % frame_skip != 0:  # Skip frames for performance boost
        continue

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        num_hands = len(results.multi_hand_landmarks)
        data_aux = []

        if num_hands == 1:
            hand_landmarks = results.multi_hand_landmarks[0]
            x_ = [lm.x for lm in hand_landmarks.landmark]
            y_ = [lm.y for lm in hand_landmarks.landmark]

            for lm in hand_landmarks.landmark:
                data_aux.append(lm.x - min(x_))
                data_aux.append(lm.y - min(y_))
            
            data_aux.extend([0] * (84 - len(data_aux)))  # Padding

            x1, y1 = int(min(x_) * W) - 10, int(min(y_) * H) - 10
            x2, y2 = int(max(x_) * W) + 10, int(max(y_) * H) + 10

        elif num_hands == 2:
            x_all, y_all = [], []
            for hand_landmarks in results.multi_hand_landmarks:
                x_ = [lm.x for lm in hand_landmarks.landmark]
                y_ = [lm.y for lm in hand_landmarks.landmark]
                for lm in hand_landmarks.landmark:
                    data_aux.append(lm.x - min(x_))
                    data_aux.append(lm.y - min(y_))
                x_all.extend(x_)
                y_all.extend(y_)
            
            x1, y1 = int(min(x_all) * W) - 10, int(min(y_all) * H) - 10
            x2, y2 = int(max(x_all) * W) + 10, int(max(y_all) * H) + 10

        # Make prediction
        try:
            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = labels_dict.get(int(prediction[0]), "Unknown")
        except ValueError:
            predicted_character = "Error"

        # Check if the sign remains the same for at least 0.8 - 1.2 sec (more flexible)
        current_time = time.time()
        if predicted_character == current_sign:
            if start_time and current_time - start_time >= 0.9:  # More flexible timing
                if predicted_character != "Unknown":
                    sentence.append(predicted_character)  # Add word to sentence
                    if len(sentence) > 5:
                        sentence.pop(0)  # Keep only last 5 words
                current_sign = ""  # Reset to detect new signs
                start_time = None
        else:
            current_sign = predicted_character
            start_time = current_time  # Start timing

        # Draw bounding box & prediction
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

    # Display last 5 words with a semi-transparent background
    displayed_text = " ".join(sentence[-5:])
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, H - 50), (W - 10, H - 10), (0, 0, 0), -1)  # Black background
    alpha = 0.5  # Transparency level
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)  # Merge overlay with frame
    cv2.putText(frame, displayed_text, (20, H - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


# """import pickle
# import cv2
# import mediapipe as mp
# import numpy as np

# # Load the model
# model_dict = pickle.load(open('./model.p', 'rb'))
# model = model_dict['model']

# # Start webcam
# cap = cv2.VideoCapture(0)

# # Set up MediaPipe Hands
# mp_hands = mp.solutions.hands
# mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles

# hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.3)

# # Label dictionary (extended for two-hand gestures if trained)
# labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E',5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',10: 'K',11: 'L',12: 'M',13: 'N',14: 'O',15: 'P',16: 'Q',17: 'R',18: 'S',19: 'T',20: 'U',21: 'V',22: 'W',23: 'X',24: 'Y',25: 'Z',27: 'TIME NOW',28: 'THANK YOU',29: 'SORRY',30: 'HELP',31: 'MOM',32: 'DAD',33: 'I LOVE U',34: 'PRAGATI',35: 'MANVENDRA',36: ''}

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("Failed to grab frame")
#         break

#     H, W, _ = frame.shape
#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#     results = hands.process(frame_rgb)
#     if results.multi_hand_landmarks:
#         # Draw landmarks for all detected hands
#         for hand_lms in results.multi_hand_landmarks:
#             mp_drawing.draw_landmarks(
#                 frame, hand_lms, mp_hands.HAND_CONNECTIONS,
#                 mp_drawing_styles.get_default_hand_landmarks_style(),
#                 mp_drawing_styles.get_default_hand_connections_style())

#         # Process hand data based on number of hands detected
#         num_hands = len(results.multi_hand_landmarks)
#         data_aux = []
        
#         if num_hands == 1:
#             # Single hand case
#             hand_landmarks = results.multi_hand_landmarks[0]
#             x_ = []
#             y_ = []
#             for i in range(len(hand_landmarks.landmark)):
#                 x = hand_landmarks.landmark[i].x
#                 y = hand_landmarks.landmark[i].y
#                 x_.append(x)
#                 y_.append(y)
#             for i in range(len(hand_landmarks.landmark)):
#                 x = hand_landmarks.landmark[i].x
#                 y = hand_landmarks.landmark[i].y
#                 data_aux.append(x - min(x_))
#                 data_aux.append(y - min(y_))

#             # Bounding box for single hand
#             x1 = int(min(x_) * W) - 10
#             y1 = int(min(y_) * H) - 10
#             x2 = int(max(x_) * W) - 10
#             y2 = int(max(y_) * H) - 10

#         elif num_hands == 2:
#             # Two-hand case: Combine data from both hands
#             x_all = []
#             y_all = []
#             for hand_landmarks in results.multi_hand_landmarks[:2]:  # Process both hands
#                 x_ = []
#                 y_ = []
#                 for i in range(len(hand_landmarks.landmark)):
#                     x = hand_landmarks.landmark[i].x
#                     y = hand_landmarks.landmark[i].y
#                     x_.append(x)
#                     y_.append(y)
#                 # Normalize per hand and append
#                 for i in range(len(hand_landmarks.landmark)):
#                     x = hand_landmarks.landmark[i].x
#                     y = hand_landmarks.landmark[i].y
#                     data_aux.append(x - min(x_))
#                     data_aux.append(y - min(y_))
#                 x_all.extend(x_)
#                 y_all.extend(y_)

#             # Bounding box encompassing both hands
#             x1 = int(min(x_all) * W) - 10
#             y1 = int(min(y_all) * H) - 10
#             x2 = int(max(x_all) * W) - 10
#             y2 = int(max(y_all) * H) - 10

#         # Make prediction
#         try:
#             prediction = model.predict([np.asarray(data_aux)])
#             predicted_character = labels_dict.get(int(prediction[0]), "Unknown")
#         except ValueError as e:
#             predicted_character = "Error: Model mismatch"
#             print(f"Prediction failed: {e}")

#         # Draw rectangle and text
#         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
#         cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)
        
#         # Indicate number of hands detected
#         cv2.putText(frame, f"Hands: {num_hands}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

#     cv2.imshow('frame', frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit with 'q'
#         break

# cap.release()
# cv2.destroyAllWindows()"""

# import pickle
# import cv2
# import mediapipe as mp
# import numpy as np

# # Load the model
# model_dict = pickle.load(open('./model.p', 'rb'))
# model = model_dict['model']

# # Start webcam
# cap = cv2.VideoCapture(0)

# # Set up MediaPipe Hands
# mp_hands = mp.solutions.hands
# mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles

# hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.3)

# labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 
#                10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 
#                19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 27: 'TIME NOW', 
#                28: 'THANK YOU', 29: 'SORRY', 30: 'HELP', 31: 'MOM', 32: 'DAD', 33: 'I LOVE U', 
#                34: 'PRAGATI', 35: 'MANVENDRA', 36: ''}

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("Failed to grab frame")
#         break

#     H, W, _ = frame.shape
#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#     results = hands.process(frame_rgb)
#     if results.multi_hand_landmarks:
#         num_hands = len(results.multi_hand_landmarks)
#         data_aux = []
        
#         if num_hands == 1:
#             hand_landmarks = results.multi_hand_landmarks[0]
#             x_ = []
#             y_ = []
#             for i in range(len(hand_landmarks.landmark)):
#                 x = hand_landmarks.landmark[i].x
#                 y = hand_landmarks.landmark[i].y
#                 x_.append(x)
#                 y_.append(y)
#             for i in range(len(hand_landmarks.landmark)):
#                 x = hand_landmarks.landmark[i].x
#                 y = hand_landmarks.landmark[i].y
#                 data_aux.append(x - min(x_))
#                 data_aux.append(y - min(y_))
#             data_aux.extend([0] * (84 - len(data_aux)))  # Pad to match two-hand input size
#             x1 = int(min(x_) * W) - 10
#             y1 = int(min(y_) * H) - 10
#             x2 = int(max(x_) * W) - 10
#             y2 = int(max(y_) * H) - 10

#         elif num_hands == 2:
#             x_all = []
#             y_all = []
#             for hand_landmarks in results.multi_hand_landmarks[:2]:
#                 x_ = []
#                 y_ = []
#                 for i in range(len(hand_landmarks.landmark)):
#                     x = hand_landmarks.landmark[i].x
#                     y = hand_landmarks.landmark[i].y
#                     x_.append(x)
#                     y_.append(y)
#                 for i in range(len(hand_landmarks.landmark)):
#                     x = hand_landmarks.landmark[i].x
#                     y = hand_landmarks.landmark[i].y
#                     data_aux.append(x - min(x_))
#                     data_aux.append(y - min(y_))
#                 x_all.extend(x_)
#                 y_all.extend(y_)
#             x1 = int(min(x_all) * W) - 10
#             y1 = int(min(y_all) * H) - 10
#             x2 = int(max(x_all) * W) - 10
#             y2 = int(max(y_all) * H) - 10

#         # Debug input size
#         print(f"Number of hands: {num_hands}, Length of data_aux: {len(data_aux)}")
        
#         # Make prediction
#         try:
#             prediction = model.predict([np.asarray(data_aux)])
#             predicted_character = labels_dict.get(int(prediction[0]), "Unknown")
#             print(f"Raw prediction: {prediction}, Predicted: {predicted_character}")
#         except ValueError as e:
#             predicted_character = "Error: Model mismatch"
#             print(f"Prediction failed: {e}")

#         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
#         cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)
#         cv2.putText(frame, f"Hands: {num_hands}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

#     cv2.imshow('frame', frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()





# import pickle
# import cv2
# import mediapipe as mp
# import numpy as np
# import time

# # Load the model
# model_dict = pickle.load(open('./model.p', 'rb'))
# model = model_dict['model']

# # Start webcam
# cap = cv2.VideoCapture(0)

# # Set up MediaPipe Hands
# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.3)

# # Label dictionary
# labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 
#                10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 
#                19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 27: 'TIME NOW', 
#                28: 'THANK YOU', 29: 'SORRY', 30: 'HELP', 31: 'MOM', 32: 'DAD', 33: 'I LOVE U', 
#                34: 'PRAGATI', 35: 'MANVENDRA', 36: ''}

# # Variables for tracking predictions
# sentence = []
# current_sign = ""
# start_time = None  # Time when the sign first appeared

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("Failed to grab frame")
#         break

#     H, W, _ = frame.shape
#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = hands.process(frame_rgb)

#     if results.multi_hand_landmarks:
#         num_hands = len(results.multi_hand_landmarks)
#         data_aux = []

#         if num_hands == 1:
#             hand_landmarks = results.multi_hand_landmarks[0]
#             x_ = [lm.x for lm in hand_landmarks.landmark]
#             y_ = [lm.y for lm in hand_landmarks.landmark]

#             for lm in hand_landmarks.landmark:
#                 data_aux.append(lm.x - min(x_))
#                 data_aux.append(lm.y - min(y_))
            
#             data_aux.extend([0] * (84 - len(data_aux)))  # Padding

#             x1, y1 = int(min(x_) * W) - 10, int(min(y_) * H) - 10
#             x2, y2 = int(max(x_) * W) + 10, int(max(y_) * H) + 10

#         elif num_hands == 2:
#             x_all, y_all = [], []
#             for hand_landmarks in results.multi_hand_landmarks:
#                 x_ = [lm.x for lm in hand_landmarks.landmark]
#                 y_ = [lm.y for lm in hand_landmarks.landmark]
#                 for lm in hand_landmarks.landmark:
#                     data_aux.append(lm.x - min(x_))
#                     data_aux.append(lm.y - min(y_))
#                 x_all.extend(x_)
#                 y_all.extend(y_)
            
#             x1, y1 = int(min(x_all) * W) - 10, int(min(y_all) * H) - 10
#             x2, y2 = int(max(x_all) * W) + 10, int(max(y_all) * H) + 10

#         # Make prediction
#         try:
#             prediction = model.predict([np.asarray(data_aux)])
#             predicted_character = labels_dict.get(int(prediction[0]), "Unknown")
#         except ValueError:
#             predicted_character = "Error"

#         # Check if the sign remains the same for at least 1 second
#         current_time = time.time()
#         if predicted_character == current_sign:
#             if start_time and current_time - start_time >= 1:  # Hold for 1 sec
#                 if predicted_character != "Unknown":
#                     sentence.append(predicted_character)  # Add word to sentence
#                     if len(sentence) > 5:
#                         sentence.pop(0)  # Keep only last 5 words
#                 current_sign = ""  # Reset to detect new signs
#                 start_time = None
#         else:
#             current_sign = predicted_character
#             start_time = current_time  # Start timing

#         # Draw bounding box & prediction
#         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
#         cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

#     # Show only last 5 words
#     displayed_text = " ".join(sentence[-5:])
#     cv2.putText(frame, displayed_text, (20, H - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

#     cv2.imshow('frame', frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()
















