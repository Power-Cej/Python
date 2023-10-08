import time
import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

# calibration factor (pixels per centimeter)
pixels_per_cm = 2

with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as holistic:

    while cap.isOpened():
        time_start = time.time()
        success, frame = cap.read()

        if not success:
            print("Frame can't read.")
            break

        frame.flags.writeable = False
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = holistic.process(frame)

        frame.flags.writeable = True
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

        if results.pose_landmarks:
            # get the frame size
            image_height, image_width, _ = frame.shape

            # identify the shoulder landmarks from left to right
            left_shoulder = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER]

            # identify hips landmarks
            right_hip = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_HIP]

            # ankle landmarks
            right_ankle = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_ANKLE]

            # calculate the length of the shoulder from left to right and convert it to cm
            shoulder_width_pixels = (
                right_shoulder.x * image_width - left_shoulder.x * image_width)
            shoulder_width_cm = abs(shoulder_width_pixels) / pixels_per_cm

            # Calculate top length
            top_length_pixel = (
                right_shoulder.y * image_height - right_hip.y * image_height)
            top_length_cm = abs(top_length_pixel) / pixels_per_cm

            # Calculate leg length
            outside_leg_height_pixels = (
                right_hip.y * image_height - right_ankle.y * image_height)
            outside_leg_length_cm = abs(
                outside_leg_height_pixels) / pixels_per_cm

            # display measurements using ladmarks detection!
            print(f'Shoulder Width (cm): {shoulder_width_cm:.2f}')
            print(f'Top length (cm): {top_length_cm:.2f}')
            print(f'Outside Leg length (cm): {outside_leg_length_cm:.2f}')

        fps = 1 / (time.time() - time_start)

        cv2.putText(frame, f'FPS: {int(fps)}', (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        cv2.rectangle(frame, (250, 15), (250 + 150, 15 + 445), (0, 255, 0), 5)

        cv2.imshow('MediaPipe Holistic', frame)

        if cv2.waitKey(5) == 27:
            break

cap.release()
cv2.destroyAllWindows()
