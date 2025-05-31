import pygame
import pymunk as pm
import pymunk.pygame_util
import math

pygame.init()
screenw=1200
screenh=678
screen=pygame.display.set_mode((screenw,screenh))
space=pm.Space()
static_body=space.static_body
draw_options=pymunk.pygame_util.DrawOptions(screen)

fps=60
clock=pygame.time.Clock()
BG=(50,50,50)

car_image=pygame.image.load("red-car-top-compact.png").convert_alpha()
car_image=pygame.transform.scale(car_image,(car_image.get_width()/2,car_image.get_height()/2))

run=True
while run:
    clock.tick(fps)
    space.step(1/fps)

    screen.fill(BG)

    screen.blit(car_image,(0,0))


    pygame.display.update()
pygame.quit()