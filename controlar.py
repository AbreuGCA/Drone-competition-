from djitellopy import Tello
import pygame
import cv2
import numpy as np
import time

import torch
from PIL import Image

tello = Tello()
tello.connect()

tello.streamon()
frame_read = tello.get_frame_read()

pygame.init()
window_width = 960
window_height = 720
screen = pygame.display.set_mode([window_width, window_height])
classes = ["Person", "Bicycle", "Car", "Motorcycle", "Airplane", "Bus", "Train", "Truck", "Boat", "Traffic light", "Fire hydrant", "Stop sign", "Parking meter", "Bench", "Bird", "Cat", "Dog", "Horse", "Sheep", "Cow", "Elephant", "Bear", "Zebra", "Giraffe", "Backpack", "Umbrella", "Handbag", "Tie", "Suitcase", "Frisbee", "Skis", "Snowboard", "Sports ball", "Kite", "Baseball bat", "Baseball glove", "Skateboard", "Surfboard", "Tennis racket", "Bottle", "Wine glass", "Cup", "Fork", "Knife", "Spoon", "Bowl", "Banana", "Apple", "Sandwich", "Orange", "Broccoli", "Carrot", "Hot dog", "Pizza", "Donut", "Cake", "Chair", "Couch", "Potted plant", "Bed", "Dining table", "Toilet", "Tv", "Laptop", "Mouse", "Remote", "Keyboard", "Cell phone", "Microwave", "Oven", "Toaster", "Sink", "Refrigerator", "Book", "Clock", "Vase", "Scissors", "Teddy bear", "Hair drier", "Toothbrush"]
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

should_stop = False
left_right = 0
up_down = 0
forward_backward = 0
yaw = 0

stop = False

velocity = 50
last_x = 0
last_y = 0
last_w = 0
last_h = 0

pygame.font.init()

texto = "Teste"
font = pygame.font.SysFont('Comic Sans MS', 30)



while not should_stop:
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_w:
                forward_backward = velocity
            if event.key == pygame.K_s:
                forward_backward = -velocity
            if event.key == pygame.K_a:
                left_right = -velocity
            if event.key == pygame.K_d:
                left_right = velocity
            if event.key == pygame.K_j:
                up_down = -velocity
            if event.key == pygame.K_k:
                up_down = velocity
            if event.key == pygame.K_h:
                yaw = -velocity
            if event.key == pygame.K_l:
                yaw = velocity
            if event.key == pygame.K_UP:
                tello.takeoff()
            if event.key == pygame.K_DOWN:
                tello.land()
            if event.key == pygame.K_f:
                tello.flip_back()
            if event.key == pygame.K_x:
                stop = True
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_w:
                forward_backward = 0
            if event.key == pygame.K_s:
                forward_backward = 0
            if event.key == pygame.K_a:
                left_right = 0
            if event.key == pygame.K_d:
                left_right = 0
            if event.key == pygame.K_j:
                up_down = 0
            if event.key == pygame.K_k:
                up_down = 0
            if event.key == pygame.K_h:
                yaw = 0
            if event.key == pygame.K_l:
                yaw = 0
        elif event.type == pygame.QUIT:
            should_stop = True
            pygame.quit()


    if frame_read.stopped:
        should_stop = True
        pygame.quit()

    frame = frame_read.frame
    #frame = np.rot90(frame)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    frame = np.rot90(frame)
    old_frame = frame
    frame = pygame.surfarray.make_surface(frame)


    results = model(old_frame)
    results = results.pred[0][results.pred[0][:, -1] == 67]
    for *xyxy, conf, cls in results:
        if conf >= 0.55:
            label = f'{classes[int(cls)]} {conf:.2f}'
            pygame.draw.rect(frame, (0, 255, 0), (int(xyxy[1]), int(xyxy[0]), int(abs(xyxy[3] - xyxy[1])), int(abs(xyxy[2] - xyxy[0]))), 1)
            #cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
            #cv2.putText(frame, label, (int(xyxy[0]), int(xyxy[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            text_surface = font.render(texto, False, (0, 0, 0))
            frame.blit(text_surface)


            coords = (xyxy[1], xyxy[0], xyxy[3], xyxy[2])
            #print(f"{corners[0][0][0]}, {corners[0][0][1]}, {corners[0][0][2]}, {corners[0][0][3]}")
            x = xyxy[1]
            y = xyxy[0]
            width  = abs(xyxy[3]-x)
            height  = abs(xyxy[2]-y)
            x = window_width -  width - x
            print(f"{x} {y} {width} {height}")

            dist = window_width/2 - x
            dist_y = window_height/2 - y

            left_right = (1 if dist > 0 else -1) * 20
            up_down = (1 if dist_y > 0 else -1) * 20

            yaw = (1 if dist > 0 else -1) * 50

            if (abs(dist) < window_height*10/100):
                up_down = 0

            if (abs(dist) < window_width*10/100):
                left_right = 0
                yaw = 0
            else:
                up_down = 0
                left_right = 0
                yaw = 0
        else:
            up_down = 0
            left_right = 0
            yaw = 0

    screen.blit(frame, (0, 0))

    if stop:
        left_right = 0
        up_down = 0
        forward_backward = 0
        yaw = 0
        velocity = 0
    if not stop:
        tello.send_rc_control(left_right, forward_backward, up_down, yaw)

    print(f"{tello.get_battery()}%")

    pygame.display.update()
    time.sleep(1 / 60)

tello.end()
