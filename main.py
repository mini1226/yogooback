import math
import mediapipe as mp
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2
import os
from flask import Flask, request, jsonify
from flask_mysqldb import MySQL
import bcrypt
import jwt
import datetime
from flask_cors import CORS


app = Flask(__name__)
CORS(app)



#---------------------------------------------------REAL TIME POSE DETECTION USING MEDIAPIPE LIBRARY-------------------------------------------

# Initializing media-pipe pose class.
mp_pose = mp.solutions.pose
# Setting up the Pose function.
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.3, model_complexity=2)
# Initializing media-pipe drawing class, useful for annotation.
mp_drawing = mp.solutions.drawing_utils


#Detect Pose
def detectPose(image, pose, display=True):
    # Create a copy of the input image.
    output_image = image.copy()

    # Convert the image from BGR into RGB format.
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Perform the Pose Detection.
    results = pose.process(imageRGB)

    # Retrieve the height and width of the input image.
    height, width, _ = image.shape

    # Initialize a list to store the detected landmarks.
    landmarks = []

    # Check if any landmarks are detected.
    if results.pose_landmarks:

        # Draw Pose landmarks on the output image.
        mp_drawing.draw_landmarks(image=output_image, landmark_list=results.pose_landmarks,
                                  connections=mp_pose.POSE_CONNECTIONS)

        # Iterate over the detected landmarks.
        for landmark in results.pose_landmarks.landmark:
            # Append the landmark into the list.
            landmarks.append((int(landmark.x * width), int(landmark.y * height),
                              (landmark.z * width)))

    # Check if the original input image and the resultant image are specified to be displayed.
    if display:

        # Display the original input image and the resultant image.
        plt.figure(figsize=[22, 22])
        plt.subplot(121);
        plt.imshow(image[:, :, ::-1]);
        plt.title("Original Image");
        plt.axis('off');
        plt.subplot(122);
        plt.imshow(output_image[:, :, ::-1]);
        plt.title("Output Image");
        plt.axis('off');

        # Also Plot the Pose landmarks in 3D.
        mp_drawing.plot_landmarks(results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)

    # Otherwise
    else:

        # Return the output image and the found landmarks.
        return output_image, landmarks


#Calculate the angle between the keypoints
def calculateAngle(landmark1, landmark2, landmark3):
    # Get the required landmarks coordinates.
    x1, y1, _ = landmark1
    x2, y2, _ = landmark2
    x3, y3, _ = landmark3

    # Calculate the angle between the three points
    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))

    # Check if the angle is less than zero.
    if angle < 0:
        # Add 360 to the found angle.
        angle += 360

    # Return the calculated angle.
    return angle


#Classify the performed pose
def classifyPose(landmarks, output_image, display=False):
    # Initialize the label of the pose. It is not known at this stage.
    label = 'Unknown Pose'
    # Specify the color (Red) with which the label will be written on the image.
    color = (0, 0, 255)

    # Calculate the required angles.
    # Get the angle between the left shoulder, elbow and wrist points.
    left_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                      landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                      landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value])

    right_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                       landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                                       landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value])

    left_shoulder_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                         landmarks[mp_pose.PoseLandmark.LEFT_HIP.value])

    right_shoulder_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                          landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value])

    left_knee_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                     landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
                                     landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value])

    right_knee_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                      landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],
                                      landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value])

    # Check if it is the warrior II pose or the T pose.
    # As for both of them, both arms should be straight and shoulders should be at the specific angle.
    # Check if the both arms are straight.
    if left_elbow_angle > 165 and left_elbow_angle < 195 and right_elbow_angle > 165 and right_elbow_angle < 195:
        # Check if shoulders are at the required angle.
        if left_shoulder_angle > 80 and left_shoulder_angle < 110 and right_shoulder_angle > 80 and right_shoulder_angle < 110:
    # Check if it is the warrior II pose.
            # Check if one leg is straight.
            if left_knee_angle > 165 and left_knee_angle < 195 or right_knee_angle > 165 and right_knee_angle < 195:
                # Check if the other leg is bent at the required angle.
                if left_knee_angle > 90 and left_knee_angle < 120 or right_knee_angle > 90 and right_knee_angle < 120:
                    # Specify the label of the pose that is Warrior II pose.
                    label = 'Warrior II Pose'


    # Check if it is the T pose.
    # Check if both legs are straight
            if left_knee_angle > 160 and left_knee_angle < 195 and right_knee_angle > 160 and right_knee_angle < 195:
                # Specify the label of the pose that is tree pose.
                label = 'T Pose'


    # Check if it is the tree pose.
    # Check if one leg is straight
    if left_knee_angle > 165 and left_knee_angle < 195 or right_knee_angle > 165 and right_knee_angle < 195:
        # Check if the other leg is bent at the required angle.
        if left_knee_angle > 315 and left_knee_angle < 335 or right_knee_angle > 25 and right_knee_angle < 45:
            # Specify the label of the pose that is tree pose.
            label = 'Tree Pose'


    # Check if the pose is classified successfully
    if label != 'Unknown Pose':
        # Update the color (to green) with which the label will be written on the image.
        color = (0, 255, 0)
    # Write the label on the output image.
    cv2.putText(output_image, label, (10, 30),cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
    # Check if the resultant image is specified to be displayed.
    if display:
        # Display the resultant image.
        plt.figure(figsize=[10,10])
        plt.imshow(output_image[:,:,::-1]);plt.title("Output Image");plt.axis('off');
    else:
        # Return the output image and the classified label.
        return output_image, label


def open_camera():
    # Setup Pose function for video.
    pose_video = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)

    # Initialize the VideoCapture object to read from the webcam.
    camera_video = cv2.VideoCapture(0)
    camera_video.set(3, 1280)
    camera_video.set(4, 960)

    # Initialize a resizable window.
    cv2.namedWindow('Pose Classification', cv2.WINDOW_NORMAL)

    # Iterate until the webcam is accessed successfully.
    while camera_video.isOpened():

        # Read a frame.
        ok, frame = camera_video.read()

        # Check if frame is not read properly.
        if not ok:
            # Continue to the next iteration to read the next frame and ignore the empty camera frame.
            continue

        # Flip the frame horizontally for natural (selfie-view) visualization.
        frame = cv2.flip(frame, 1)

        # Get the width and height of the frame
        frame_height, frame_width, _ = frame.shape

        # Resize the frame while keeping the aspect ratio.
        frame = cv2.resize(frame, (int(frame_width * (640 / frame_height)), 640))

        # Perform Pose landmark detection.
        frame, landmarks = detectPose(frame, pose_video, display=False)

        # Check if the landmarks are detected.
        if landmarks:
            # Perform the Pose Classification.
            frame, _ = classifyPose(landmarks, frame, display=False)

        # Display the frame.
        cv2.imshow('Pose Classification', frame)

        # Wait until a key is pressed.
        # Retreive the ASCII code of the key pressed
        k = cv2.waitKey(1) & 0xFF

        # Check if 'ESC' is pressed.
        if (k == 27):
            # Break the loop.
            break

    # Release the VideoCapture object and close the windows.
    camera_video.release()
    cv2.destroyAllWindows()


@app.route('/open_camera')
def api_open_camera():
    open_camera()
    return 'Camera closed'




# --------------------------------------------------------------GUIDE FOR POSES-----------------------------------------------------------------

#--------------------------------------------Guide Tpose-----------------------------------------------------------
def guideTpose(landmarks, output_image, display=False):
    # Initialize the label of the pose. It is not known at this stage.
    label = 'Unknown Pose'
    # Specify the color (Red) with which the label will be written on the image.
    color = (0, 255, 0)

    # Calculate the required angles.
    # Get the angle between the left shoulder, elbow and wrist points.
    left_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                      landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                      landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value])

    right_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                       landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                                       landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value])

    left_shoulder_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                         landmarks[mp_pose.PoseLandmark.LEFT_HIP.value])

    right_shoulder_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                          landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value])

    left_knee_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                     landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
                                     landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value])

    right_knee_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                      landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],
                                      landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value])


    # Check if the both arms are straight.
    if left_elbow_angle > 165 and left_elbow_angle < 195 and right_elbow_angle > 165 and right_elbow_angle < 195:
        # Check if shoulders are at the required angle.
        if left_shoulder_angle > 80 and left_shoulder_angle < 110 and right_shoulder_angle > 80 and right_shoulder_angle < 110:
            #Check if both legs are straight
            if left_knee_angle > 160 and left_knee_angle < 195 and right_knee_angle > 160 and right_knee_angle < 195:
                # Specify the label of the pose that is tree pose.
                label = 'T Pose'
            else:
                label = 'legs should be straight'
        else:
            label = 'Extend arms to the sides'
    else:
        label = 'both arms should be straight'


    # Check if the pose is classified successfully
    if label != 'T Pose':
        # Update the color (to green) with which the label will be written on the image.
        color = (0, 0, 255)
    # Write the label on the output image.
    cv2.putText(output_image, label, (10, 30),cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
    # Check if the resultant image is specified to be displayed.
    if display:
        # Display the resultant image.
        plt.figure(figsize=[10,10])
        plt.imshow(output_image[:,:,::-1]);plt.title("Output Image");plt.axis('off');
    else:
        # Return the output image and the classified label.
        return output_image, label



#---------------------------------------------Guide Treepose--------------------------------------------------------
def guideTreePose(landmarks, output_image, display=False):
    # Initialize the label of the pose. It is not known at this stage.
    label = 'Unknown Pose'
    # Specify the color (Red) with which the label will be written on the image.
    color = (0, 255, 0)

    # Calculate the required angles.
    # Get the angle between the left shoulder, elbow and wrist points.
    left_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                      landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                      landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value])

    right_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                       landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                                       landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value])

    left_shoulder_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                         landmarks[mp_pose.PoseLandmark.LEFT_HIP.value])

    right_shoulder_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                          landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value])

    left_knee_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                     landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
                                     landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value])

    right_knee_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                      landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],
                                      landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value])

    # Check if the both arms are straight.
    if left_elbow_angle > 150 and left_elbow_angle < 195 and right_elbow_angle > 150 and right_elbow_angle < 195:
        # Check if shoulders are at the required angle.
        if left_shoulder_angle > 160 and left_shoulder_angle < 195 and right_shoulder_angle > 160 and right_shoulder_angle < 195:
            # Check if one leg is straight
            if left_knee_angle > 165 and left_knee_angle < 195 or right_knee_angle > 165 and right_knee_angle < 195:
                #Check if the other leg is bended at the required angle.
                if left_knee_angle > 315 and left_knee_angle < 335 or right_knee_angle > 25 and right_knee_angle < 45:
                    # Specify the label of the pose that is tree pose.
                    label = 'Tree Pose'
                else:
                    label = 'other leg should be bended'
            else:
                label = 'One leg should be straight'
        else:
            label = 'Extend arms towards head '
    else:
        label = 'both arms should be straight'

    # Check if the pose is classified successfully
    if label != 'Tree Pose':
        # Update the color (to green) with which the label will be written on the image.
        color = (0, 0, 255)
    # Write the label on the output image.
    cv2.putText(output_image, label, (10, 30),cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
    # Check if the resultant image is specified to be displayed.
    if display:
        # Display the resultant image.
        plt.figure(figsize=[10,10])
        plt.imshow(output_image[:,:,::-1]);plt.title("Output Image");plt.axis('off');
    else:
        # Return the output image and the classified label.
        return output_image, label


#--------------------------------------------Warrior II Pose--------------------------------------------------------
def guideWarriorIIPose(landmarks, output_image, display=False):
    # Initialize the label of the pose. It is not known at this stage.
    label = 'Unknown Pose'
    # Specify the color (Red) with which the label will be written on the image.
    color = (0, 255, 0)

    # Calculate the required angles.
    # Get the angle between the left shoulder, elbow and wrist points.
    left_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                      landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                      landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value])

    right_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                       landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                                       landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value])

    left_shoulder_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                         landmarks[mp_pose.PoseLandmark.LEFT_HIP.value])

    right_shoulder_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                          landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value])

    left_knee_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                     landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
                                     landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value])

    right_knee_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                      landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],
                                      landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value])


    # Check if the both arms are straight.
    if left_elbow_angle > 165 and left_elbow_angle < 195 and right_elbow_angle > 165 and right_elbow_angle < 195:
        # Check if shoulders are at the required angle.
        if left_shoulder_angle > 80 and left_shoulder_angle < 110 and right_shoulder_angle > 80 and right_shoulder_angle < 110:
            # Check if one leg is straight.
            if left_knee_angle > 165 and left_knee_angle < 195 or right_knee_angle > 165 and right_knee_angle < 195:
                # Check if the other leg is bent at the required angle.
                if left_knee_angle > 90 and left_knee_angle < 120 or right_knee_angle > 90 and right_knee_angle < 120:
                    # Specify the label of the pose that is Warrior II pose.
                    label = 'Warrior II Pose'
                else:
                    label = 'other leg should be bend'
            else:
                label = 'one leg should be straight'
        else:
            label = 'Extend arms to the sides'
    else:
        label = 'both arms should be straight'


    # Check if the pose is classified successfully
    if label != 'Warrior II Pose':
        # Update the color (to green) with which the label will be written on the image.
        color = (0, 0, 255)
    # Write the label on the output image.
    cv2.putText(output_image, label, (10, 30),cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
    # Check if the resultant image is specified to be displayed.
    if display:
        # Display the resultant image.
        plt.figure(figsize=[10,10])
        plt.imshow(output_image[:,:,::-1]);plt.title("Output Image");plt.axis('off');
    else:
        # Return the output image and the classified label.
        return output_image, label




#--------------------------------------------Dog Pose--------------------------------------------------------
def guideDogPose(landmarks, output_image, display=False):
    # Initialize the label of the pose. It is not known at this stage.
    label = 'Unknown Pose'
    # Specify the color (Red) with which the label will be written on the image.
    color = (0, 255, 0)

    # Calculate the required angles.
    # Get the angle between the left shoulder, elbow and wrist points.
    left_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                      landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                      landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value])

    right_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                       landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                                       landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value])

    left_shoulder_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                         landmarks[mp_pose.PoseLandmark.LEFT_HIP.value])

    right_shoulder_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                          landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value])

    left_knee_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                     landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
                                     landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value])

    right_knee_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                      landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],
                                      landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value])

    left_hip_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
                                    landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                    landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value])

    right_hip_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                     landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                     landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value])

    # Check if it is the dog pose.
    # ----------------------------------------------------------------------------------------------------------------

    # Check if the both arms are straight.
    if left_elbow_angle > 165 and left_elbow_angle < 195 and right_elbow_angle > 165 and right_elbow_angle < 195:
        #Check if shoulders are at the required angle.
        if left_shoulder_angle > 160 and left_shoulder_angle < 195 and right_shoulder_angle > 160 and right_shoulder_angle < 195:
            # Check if the legs is bended at the required angle.
            if left_knee_angle > 160 and left_knee_angle < 195 or right_knee_angle > 160 and right_knee_angle < 195:
                # Check if the both arms are straight.
                if left_hip_angle > 30 and left_hip_angle < 70 or right_hip_angle > 30 and right_hip_angle < 70:
                    # Specify the label of the pose that is tree pose.
                    label = 'Dog Pose'
                else:
                    label = 'body should bend from the hip'
            else:
                label = 'both legs should be straight'
        else:
            label = 'Extend arms towards head '
    else:
        label = 'both arms should be straight'


#----------------------------------------------------------------------------------------------------------------

    # Check if the pose is classified successfully
    if label != 'Dog Pose':
        # Update the color (to green) with which the label will be written on the image.
        color = (0, 0, 255)
    # Write the label on the output image.
    cv2.putText(output_image, label, (10, 30),cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
    # Check if the resultant image is specified to be displayed.
    if display:
        # Display the resultant image.
        plt.figure(figsize=[10,10])
        plt.imshow(output_image[:,:,::-1]);plt.title("Output Image");plt.axis('off');
    else:
        # Return the output image and the classified label.
        return output_image, label




#--------------------------------------------Warrior Pose--------------------------------------------------------
def guideWarriorPose(landmarks, output_image, display=False):
    # Initialize the label of the pose. It is not known at this stage.
    label = 'Unknown Pose'
    # Specify the color (Red) with which the label will be written on the image.
    color = (0, 255, 0)

    # Calculate the required angles.
    # Get the angle between the left shoulder, elbow and wrist points.
    left_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                      landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                      landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value])

    right_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                       landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                                       landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value])

    left_shoulder_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                         landmarks[mp_pose.PoseLandmark.LEFT_HIP.value])

    right_shoulder_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                          landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value])

    left_knee_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                     landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
                                     landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value])

    right_knee_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                      landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],
                                      landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value])


    # Check if it is the warrior pose.
    # ----------------------------------------------------------------------------------------------------------------

    # Check if it is the warrior pose.
    # ----------------------------------------------------------------------------------------------------------------

    # Check if the both arms are straight.
    if left_elbow_angle > 165 and left_elbow_angle < 195 and right_elbow_angle > 165 and right_elbow_angle < 195:
        # Check if shoulders are at the required angle.
        if left_shoulder_angle > 160 and left_shoulder_angle < 195 and right_shoulder_angle > 160 and right_shoulder_angle < 195:
            # Check if the legs is bent at the required angle.
            if left_knee_angle > 160 and left_knee_angle < 195 or right_knee_angle > 160 and right_knee_angle < 195:
                # Check if the other leg is bent at the required angle.
                if left_knee_angle > 90 and left_knee_angle < 120 or right_knee_angle > 90 and right_knee_angle < 120:
                    # Specify the label of the pose that is Warrior II pose.
                    label = 'Warrior Pose'
                else:
                    label = 'other leg should be bend'
            else:
                label = 'one leg should be straight'
        else:
            label = 'Extend arms towards head '
    else:
        label = 'both arms should be straight'


#----------------------------------------------------------------------------------------------------------------

    # Check if the pose is classified successfully
    if label != 'Warrior Pose':
        # Update the color (to green) with which the label will be written on the image.
        color = (0, 0, 255)
    # Write the label on the output image.
    cv2.putText(output_image, label, (10, 30),cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
    # Check if the resultant image is specified to be displayed.
    if display:
        # Display the resultant image.
        plt.figure(figsize=[10,10])
        plt.imshow(output_image[:,:,::-1]);plt.title("Output Image");plt.axis('off');
    else:
        # Return the output image and the classified label.
        return output_image, label




#--------------------------------------------Triangle Pose--------------------------------------------------------
def guideTrianglePose(landmarks, output_image, display=False):
    # Initialize the label of the pose. It is not known at this stage.
    label = 'Unknown Pose'
    # Specify the color (Red) with which the label will be written on the image.
    color = (0, 255, 0)

    # Calculate the required angles.
    # Get the angle between the left shoulder, elbow and wrist points.
    left_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                      landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                      landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value])

    right_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                       landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                                       landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value])

    left_shoulder_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                         landmarks[mp_pose.PoseLandmark.LEFT_HIP.value])

    right_shoulder_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                          landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value])

    left_knee_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                     landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
                                     landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value])

    right_knee_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                      landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],
                                      landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value])

    left_hip_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
                                    landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                    landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value])

    right_hip_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                     landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                     landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value])


    # Check if it is the triangle pose.
    # ----------------------------------------------------------------------------------------------------------------

    # Check if it is the triangle pose.
    # ----------------------------------------------------------------------------------------------------------------

    # Check if the both arms are straight.
    if left_elbow_angle > 165 and left_elbow_angle < 195 and right_elbow_angle > 165 and right_elbow_angle < 195:
        # Check if shoulders are at the required angle.
        if left_shoulder_angle > 80 and left_shoulder_angle < 110 and right_shoulder_angle > 80 and right_shoulder_angle < 110:
        #Check if the legs is bent at the required angle.
            if left_knee_angle > 160 and left_knee_angle < 195 and right_knee_angle > 160 and right_knee_angle < 195:
                # Check if the both arms are straight.
                if left_hip_angle > 30 and left_hip_angle < 70 or right_hip_angle > 30 and right_hip_angle < 70:
                    # Specify the label of the pose that is tree pose.
                    label = 'Triangle Pose'
                else:
                    label = 'Bend a side from the hip'
            else:
                label = 'Both legs should be straight'
        else:
            label = 'Extend arms to the sides'
    else:
        label = 'both arms should be straight'

    #----------------------------------------------------------------------------------------------------------------

    # Check if the pose is classified successfully
    if label != 'Dog Pose':
        # Update the color (to green) with which the label will be written on the image.
        color = (0, 0, 255)
    # Write the label on the output image.
    cv2.putText(output_image, label, (10, 30),cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
    # Check if the resultant image is specified to be displayed.
    if display:
        # Display the resultant image.
        plt.figure(figsize=[10,10])
        plt.imshow(output_image[:,:,::-1]);plt.title("Output Image");plt.axis('off');
    else:
        # Return the output image and the classified label.
        return output_image, label






# Guide API
@app.route('/guide', methods=['POST'])
def guide_pose():
    # Get user input from request body
    pose = request.json['pose']

    # Setup Pose function for video.
    pose_video = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)

    # Initialize the VideoCapture object to read from the webcam.
    camera_video = cv2.VideoCapture(0)
    camera_video.set(3, 1280)
    camera_video.set(4, 960)

    # Initialize a resizable window.
    cv2.namedWindow('Pose Classification', cv2.WINDOW_NORMAL)

    # Iterate until the webcam is accessed successfully.
    while camera_video.isOpened():

        # Read a frame.
        ok, frame = camera_video.read()

        # Check if frame is not read properly.
        if not ok:
            # Continue to the next iteration to read the next frame and ignore the empty camera frame.
            continue

        # Flip the frame horizontally for natural (selfie-view) visualization.
        frame = cv2.flip(frame, 1)

        # Get the width and height of the frame
        frame_height, frame_width, _ = frame.shape

        # Resize the frame while keeping the aspect ratio.
        frame = cv2.resize(frame, (int(frame_width * (640 / frame_height)), 640))

        # Perform Pose landmark detection.
        frame, landmarks = detectPose(frame, pose_video, display=False)

        # Check if the landmarks are detected.
        if landmarks and pose == 'TPose':
            # Perform the Pose Classification.
            frame, _ = guideTpose(landmarks, frame, display=False)
        elif landmarks and pose == 'TreePose':
            # Perform the Pose Classification.
            frame, _ = guideTreePose(landmarks, frame, display=False)
        elif landmarks and pose == 'WarriorIIPose':
            # Perform the Pose Classification.
            frame, _ = guideWarriorIIPose(landmarks, frame, display=False)
        elif landmarks and pose == 'DogPose':
            # Perform the Pose Classification.
            frame, _ = guideDogPose(landmarks, frame, display=False)
        elif landmarks and pose == 'WarriorPose':
            # Perform the Pose Classification.
            frame, _ = guideWarriorPose(landmarks, frame, display=False)
        elif landmarks and pose == 'TrianglePose':
            # Perform the Pose Classification.
            frame, _ = guideWarriorPose(landmarks, frame, display=False)
        else:
            pose = 'no pose mentioned'
        # Display the frame.
        cv2.imshow('Pose Classification', frame)

        # Wait until a key is pressed.
        # Retreive the ASCII code of the key pressed
        k = cv2.waitKey(1) & 0xFF

        # Check if 'ESC' is pressed.
        if (k == 27):
            # Break the loop.
            break

    # Release the VideoCapture object and close the windows.
    camera_video.release()
    cv2.destroyAllWindows()
    return pose




#---------------------------------------------------------------FILE UPLOADING API-------------------------------------------------------------

app.config['VIDEO_UPLOAD_FOLDER'] = './uploads/videos'

ALLOWED_EXTENSIONS_VIDEO = {'mp4'}

def allowed_video_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS_VIDEO

@app.route('/upload', methods=['POST'])
def upload_video_file():
    try:
        file = request.files['file']
        if file and allowed_video_file(file.filename):
            file.save(os.path.join(app.config['VIDEO_UPLOAD_FOLDER'], file.filename))
            return jsonify({'message': 'File uploaded successfully'}), 200
        else:
            return jsonify({'error': 'Invalid file extension. Only mp4 files are allowed.'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 400




#---------------------------------------------------------------POSE CLASSIFICATION-------------------------------------------------------------
IMAGE_HEIGHT , IMAGE_WIDTH = 128, 128
SEQUENCE_LENGTH = 200
CLASSES_LIST = ['Padamasana', 'Tadasana', 'Vrikshasana', 'Trikasana', 'Bhujasana']

LRCN_model=tf.keras.models.load_model("LRCN_model___Date_Time_2023_04_21__00_45_13___Loss_0.6665855646133423___Accuracy_0.8333333134651184.h5")


def predict_single_action(video_file_path, SEQUENCE_LENGTH):
    '''
    This function will perform single action recognition prediction on a video using the LRCN model.
    Args:
    video_file_path:  The path of the video stored in the disk on which the action recognition is to be performed.
    SEQUENCE_LENGTH:  The fixed number of frames of a video that can be passed to the model as one sequence.
    '''

    # Initialize the VideoCapture object to read from the video file.
    video_reader = cv2.VideoCapture(video_file_path)

    # Get the width and height of the video.
    original_video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Declare a list to store video frames we will extract.
    frames_list = []

    # Initialize a variable to store the predicted action being performed in the video.
    predicted_class_name = ''

    # Get the number of frames in the video.
    video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate the interval after which frames will be added to the list.
    skip_frames_window = max(int(video_frames_count / SEQUENCE_LENGTH), 1)

    # Iterating the number of times equal to the fixed length of sequence.
    for frame_counter in range(SEQUENCE_LENGTH):

        # Set the current frame position of the video.
        video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window)

        # Read a frame.
        success, frame = video_reader.read()

        # Check if frame is not read properly then break the loop.
        if not success:
            break

        # Resize the Frame to fixed Dimensions.
        resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))

        # Normalize the resized frame by dividing it with 255 so that each pixel value then lies between 0 and 1.
        normalized_frame = resized_frame / 255

        # Appending the pre-processed frame into the frames list
        frames_list.append(normalized_frame)

    # Passing the  pre-processed frames to the model and get the predicted probabilities.
    predicted_labels_probabilities = LRCN_model.predict(np.expand_dims(frames_list, axis=0))[0]

    # Get the index of class with the highest probability.
    predicted_label = np.argmax(predicted_labels_probabilities)

    # Get the class name using the retrieved index.
    predicted_class_name = CLASSES_LIST[predicted_label]

    # Display the predicted action along with the prediction confidence.
    print(f'Action Predicted: {predicted_class_name}\nConfidence: {predicted_labels_probabilities[predicted_label]}')

    # Release the VideoCapture object.
    video_reader.release()

    if (np.argmax(predicted_labels_probabilities) < 0.5):
        return 'Wrong pose performed !'
    else:
        return predicted_class_name


@app.route('/classify', methods=['POST'])
def api():
    data = request.get_json()
    video_title = data.get('name')

    # Construct the input YouTube video path
    input_video_file_path = 'uploads/videos/' + video_title

    # Perform Single Prediction on the Test Video.
    class_name = predict_single_action(input_video_file_path, SEQUENCE_LENGTH)

    # result = {'name': name}
    return jsonify(class_name)



#---------------------------------------------------------------AUTHENTICATION-------------------------------------------------------------
# Configure MySQL
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'yoogah'

# Initialize MySQL
mysql = MySQL(app)

# Define a secret key for JWT
app.config['SECRET_KEY'] = 'mysecretkey'

# Registration API
@app.route('/register', methods=['POST'])
def register():
    try:
        # Get user input from request body
        firstName = request.json['firstName']
        lastName = request.json['lastName']
        email = request.json['email']
        password = request.json['password']

        # Check if user already exists
        cur = mysql.connection.cursor()
        cur.execute("SELECT * FROM users WHERE email = %s", [email])
        user = cur.fetchone()
        cur.close()

        if user:
            # Return error message if user already exists
            return jsonify({'message': 'User already exists.'}), 400

        # Hash the password
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

        # Save user details to database
        cur = mysql.connection.cursor()
        cur.execute("INSERT INTO users (firstName, lastName, email, password) VALUES (%s, %s, %s, %s)", (firstName, lastName, email, hashed_password))
        mysql.connection.commit()
        cur.close()

        # Return success message
        return jsonify({'message': 'User registered successfully.'}), 201
    except Exception as e:
        # Handle any exceptions that may occur during the registration process
        return jsonify({'message': str(e)}), 500




# Login API
@app.route('/login', methods=['POST'])
def login():
    try:
        # Get user input from request body
        email = request.json['email']
        password = request.json['password']

        # Check if user exists
        cur = mysql.connection.cursor()
        cur.execute("SELECT * FROM users WHERE email = %s", [email])
        user = cur.fetchone()
        cur.close()

        if user:
            # Check if password is correct
            if bcrypt.checkpw(password.encode('utf-8'), user[4].encode('utf-8')):
                # Generate JWT token
                token = jwt.encode({'user_id': user[0], 'exp': datetime.datetime.utcnow() + datetime.timedelta(minutes=30)}, app.config['SECRET_KEY'])

                # Return token as response
                return jsonify({'token': token,
                                'userId': user[0],
                                'firstName': user[1],
                                'lastName': user[2],
                                'email': user[3]})

        # Return error message if authentication fails
        return jsonify({'message': 'Invalid credentials.'}), 401
    except Exception as e:
        # Handle any exceptions that may occur during the registration process
        return jsonify({'message': str(e)}), 500



#-----------------------------------------------------------MultiPose-------------------------------------------------------------------


#-----------------------------------------------------------Img uploads for multi----------------------------------------------------------
app.config['IMAGE_UPLOAD_FOLDER'] = './uploads/images'

ALLOWED_EXTENSIONS_IMAGES = {'jpg'}

def allowed_image_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS_IMAGES

@app.route('/imgupload', methods=['POST'])
def upload_image_file():
    try:
        file = request.files['file']
        if file and allowed_image_file(file.filename):
            file.save(os.path.join(app.config['IMAGE_UPLOAD_FOLDER'], file.filename))
            return jsonify({'message': 'File uploaded successfully'}), 200
        else:
            return jsonify({'error': 'Invalid file extension. Only jpg files are allowed.'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 400




def classifyImagePose(landmarks, output_image, display=False):
    # Initialize the label of the pose. It is not known at this stage.
    label = 'Unknown Pose'
    # Specify the color (Red) with which the label will be written on the image.
    color = (0, 0, 255)
    # Calculate the required angles.
    # Get the angle between the left shoulder, elbow and wrist points.
    left_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                      landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                      landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value])
    right_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                       landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                                       landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value])
    left_shoulder_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                         landmarks[mp_pose.PoseLandmark.LEFT_HIP.value])
    right_shoulder_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                          landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value])
    left_knee_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                     landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
                                     landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value])
    right_knee_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                      landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],
                                      landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value])
    # Check if it is the warrior II pose or the T pose.
    # As for both of them, both arms should be straight and shoulders should be at the specific angle.
    # Check if the both arms are straight.
    if left_elbow_angle > 165 and left_elbow_angle < 195 and right_elbow_angle > 165 and right_elbow_angle < 195:
        # Check if shoulders are at the required angle.
        if left_shoulder_angle > 80 and left_shoulder_angle < 110 and right_shoulder_angle > 80 and right_shoulder_angle < 110:
    # Check if it is the warrior II pose.
            # Check if one leg is straight.
            if left_knee_angle > 165 and left_knee_angle < 195 or right_knee_angle > 165 and right_knee_angle < 195:
                # Check if the other leg is bended at the required angle.
                if left_knee_angle > 90 and left_knee_angle < 120 or right_knee_angle > 90 and right_knee_angle < 120:
                    # Specify the label of the pose that is Warrior II pose.
                    label = 'Warrior II Pose'
    # Check if it is the T pose.
    # Check if both legs are straight
            if left_knee_angle > 160 and left_knee_angle < 195 and right_knee_angle > 160 and right_knee_angle < 195:
                # Specify the label of the pose that is tree pose.
                label = 'T Pose'
    # Check if it is the tree pose.
    # Check if one leg is straight
    if left_knee_angle > 165 and left_knee_angle < 195 or right_knee_angle > 165 and right_knee_angle < 195:
        # Check if the other leg is bended at the required angle.
        if left_knee_angle > 315 and left_knee_angle < 335 or right_knee_angle > 25 and right_knee_angle < 45:
            # Specify the label of the pose that is tree pose.
            label = 'Tree Pose'
    # Check if the pose is classified successfully
    if label != 'Unknown Pose':
        # Update the color (to green) with which the label will be written on the image.
        color = (0, 255, 0)
    return label



#-------------------------------------------------split uploaded image------------------------------------------------------------------------

# @app.route('/split_image', methods=['POST'])
# def split_image():
#     data = request.get_json()
#     image_title = data.get('name')
#     num_splits = data.get('num_splits')
#
#     # Construct the input YouTube video path
#     input_image_file_path = 'uploads/images/' + image_title
#
#     img = cv2.imread(input_image_file_path)
#
#     # Calculate the height and width of each split
#     height, width, _ = img.shape
#     split_height = height
#     split_width = math.floor(width / num_splits)
#
#     # Split the image into splits
#     split_names = []
#     for i in range(num_splits):
#         split = img[0:split_height, i * split_width : (i + 1) * split_width]
#         filename = f'uploads/images/{image_title}_{i}.jpg'
#         cv2.imwrite(filename, split)
#         split_names.append(filename)
#
#     # for i in range(num_splits):
#     #     filename = f'uploads/images/{image_title}_{i}.jpg'
#     #     print(filename)
#
#     # Return the names of the split images in a JSON response
#     response = {
#         'split_names': split_names
#     }
#     return jsonify(response)



import base64

@app.route('/split_image', methods=['POST'])
def split_image():
    data = request.get_json()
    image_title = data.get('name')
    num_splits = data.get('num_splits')

    # Construct the input YouTube video path
    input_image_file_path = 'uploads/images/' + image_title
    output = 'pose not detected'

    img = cv2.imread(input_image_file_path)

    # Calculate the height and width of each split
    height, width, _ = img.shape
    split_height = height
    split_width = math.floor(width / num_splits)

    # Split the image into splits
    split_data = []
    for i in range(num_splits):
        split = img[0:split_height, i * split_width : (i + 1) * split_width]
        _, encoded_image = cv2.imencode('.jpg', split)
        base64_image = base64.b64encode(encoded_image).decode('utf-8')

        filename = f'uploads/images/{image_title}_{i}.jpg'
        cv2.imwrite(filename, split)

        # Read a sample image and perform classification on it.
        image = cv2.imread(filename)
        output_image, landmarks = detectPose(image, pose, display=False)
        if landmarks:
            output = classifyImagePose(landmarks, output_image, display=True)

        split_data.append({
            'name': filename,
            'data': base64_image,
            'pose': output
        })

    # Return the split image data in a JSON response
    response = {
        'splits': split_data
    }
    return jsonify(response)



@app.route('/imgmulti', methods=['POST'])
def multiapi():
    data = request.get_json()
    image_title = data.get('name')

    output = 'pose not detected'

    # Construct the input YouTube video path
    input_image_file_path = 'uploads/images/' + image_title

    # Read a sample image and perform classification on it.
    image = cv2.imread(input_image_file_path)
    output_image, landmarks = detectPose(image, pose, display=False)
    if landmarks:
        output = classifyImagePose(landmarks, output_image, display=True)

    return output





if __name__ == '__main__':
    app.run()
