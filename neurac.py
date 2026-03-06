from neural_network import *

from copy import deepcopy
from random import randint, gauss, choice
from threading import Thread
from time import time, sleep

import math
import pygame

load_neural = True
save_neural = False
draw_tracers = True

with_player = False
speed = 2
cars_number = 10

def trace(x, y, angle):
    wx, wy = x, y
    radians = angle * math.pi / 180
    dx = math.cos(radians) * 5
    dy = math.sin(radians) * 5

    for i in range(50):
        if win.get_at((int(x), int(y))) == border:
            break

        x += dx
        y += dy

    d = ((x - wx) ** 2 + (y - wy) ** 2) ** 0.5
    return int(x), int(y), d

class Car:
    def __init__(self):
        self.tracers = []
        self.network = NeuralNetwork(6, [24, 12, 2], ['relu', 'relu', 'tanh'])
        if load_neural:
            self.network.load('RacingNeural2')
            self.dead = True

        self.surface = pygame.Surface((16, 10), pygame.SRCALPHA, 32)
        self.surface = self.surface.convert_alpha()

        self.restart()

    def restart(self):
        self.x = 400
        self.y = 75

        self.angle = 0
        self.vel = 0

        self.dead = False
        self.score = 0
        self.was_score = 0

        self.color = (255, 25, 25)
        self.update()

    def update(self):
        self.surface.fill(self.color)

    def trace(self):
        angle = (self.angle + 270) % 360
        self.tracers = []
        for i in range(5):
            self.tracers.append(trace(self.x + 8, self.y + 5, angle))
            angle = (angle + 45) % 360

    def draw(self):
        if not self.dead or self.color == (25, 25, 255):
            if draw_tracers:
                for t in self.tracers:
                    pygame.draw.line(win, (150, 200, 25), (int(self.x) + 12, int(self.y) + 8), (t[0], t[1]))
            surface = pygame.transform.rotate(self.surface, -self.angle)
            win.blit(surface, (int(self.x), int(self.y)))

    def work(self):
        if self.dead:
            return

        self.trace()
        result = self.network.feed([t[2] for t in self.tracers] + [self.vel])

        self.vel += result[0] / 2 + 0.25
        self.vel = min(self.vel, speed)
        self.vel = max(self.vel, -speed / 5)
        self.angle += (3 * result[1]) * (self.vel / 2.5)

        radians = self.angle * math.pi / 180
        self.x += math.cos(radians) * self.vel
        self.y += math.sin(radians) * self.vel

        if win.get_at((int(self.x) + 8, int(self.y) + 5)) == border:
            self.dead = True

        if self.score < 5000:
            self.score += self.vel
            if self.x < 100 and self.y > 400:
                print(frame, 'Frames')
                self.score = 10000 - time() + gentime

class Player:
    def __init__(self):
        self.restart()

        self.color = (25, 255, 25)

        self.surface = pygame.Surface((16, 10), pygame.SRCALPHA, 32)
        self.surface = self.surface.convert_alpha()
        self.surface.fill(self.color)

    def restart(self):
        self.x = 400
        self.y = 75

        self.angle = 0
        self.vel = 0

    def draw(self):
        surface = pygame.transform.rotate(self.surface, -self.angle)
        win.blit(surface, (int(self.x), int(self.y)))

    def work(self):
        keys = pygame.key.get_pressed()

        if keys[pygame.K_s]:
            self.vel -= 0.05
        elif keys[pygame.K_w]:
            self.vel += 0.75

        if keys[pygame.K_a]:
            self.angle -= self.vel * 1.2
        if keys[pygame.K_d]:
            self.angle += self.vel * 1.2

        self.vel = min(self.vel, speed)
        self.vel = max(self.vel, -speed / 5)

        radians = self.angle * math.pi / 180
        self.x += math.cos(radians) * self.vel
        self.y += math.sin(radians) * self.vel

def loop_thread(part):
    for car in part:
        car.work()

pygame.init()
pygame.font.init()

font = pygame.font.SysFont('Comic Sans MS', 15)
road = pygame.image.load('race1.png')
rect = road.get_rect()

win = pygame.display.set_mode((rect[2], rect[3]), pygame.DOUBLEBUF)
win.set_alpha(None)

#sleep(10)

road = road.convert()
pygame.display.set_caption('Racing Neural Netwrok')

clock = pygame.time.Clock()

border = (0, 0, 0, 255)
cars = []
for i in range(cars_number):
    cars.append(Car())
cars[0].dead = False

if with_player:
    player = Player()

gentime = time()
bestcars = []
next = load_neural

generation = 1
frame = 0

running = True
while running:
    frame += 1
    time_ = time()
    clock.tick(60)

    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            running = False
        elif e.type == pygame.KEYUP and e.key == pygame.K_ESCAPE:
            running = False
        elif e.type == pygame.MOUSEBUTTONUP:
            if e.button == 3:
                next = True

    if next and bestcars:
        print('---------------')
        frame = 0

        generation += 1
        next = False
        gentime = time()

        if save_neural:
            bestcars[-1].network.save('RacingNeural2')

        for car in cars:
            car.restart()
            if car in bestcars:
                continue

            car.network.layers = deepcopy(choice(bestcars).network.layers)

            for layer in car.network.layers:
                for neuron in layer:
                    for weight in range(len(neuron.weights)):
                        if randint(0, 10) == 0:
                            neuron.weights[weight] += gauss(0, 1) * 0.5

                    if randint(0, 10) == 0:
                        neuron.bias += gauss(0, 1) * 0.5
        bestcars = []

    win.blit(road, (0, 0))

    threads = []
    for i in range(10):
        part = cars[i * 10: (i + 1) * 10]
        thread = Thread(target = loop_thread, args = (part, ), daemon = True)

        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    for car in cars:
        if car not in bestcars:
            car.draw()
    for car in bestcars:
        car.draw()

    if with_player:
        player.work()
        if win.get_at((int(player.x) + 8, int(player.y) + 5)) == border:
            player.restart()
            next = True

        player.draw()

    fps = font.render(str(int(1 / (time() - time_))), False, (150, 25, 25))
    gen = font.render('Generation %s' % generation, False, (150, 150, 25))

    win.blit(fps, (2, 2))
    win.blit(gen, (2, 22))

    pygame.display.flip()

    if frame % 10 == 0:
        if bestcars:
            for car in bestcars:
                car.color = (255, 25, 25)
                car.update()

        cars.sort(key = lambda car: car.score)
        bestcars = cars.copy()[-4:]
        if bestcars[0].score > 5000:
            next = True

        for car in bestcars:
            car.color = (25, 25, 255)
            car.update()

    if frame % 60 == 0:
        cars.sort(key = lambda car: car.score - car.was_score)
        max_delta = cars[-1].score - cars[-1].was_score

        for car in cars:
            car.was_score = car.score

        if max_delta < 10 and time() - gentime > 2 and not load_neural or time() - gentime > 30:
            next = True

pygame.quit()
