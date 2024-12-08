import pygame

def main():
    pygame.init()

    window = pygame.display.set_mode((500, 500))

    pygame.display.set_caption('Pygame Basic Movement and Key Presses')

    x = 50
    y = 50
    width = 40
    height = 60
    velocity = 5

    run = True
    while run:
        pygame.time.delay(100)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            y -= velocity
        if keys[pygame.K_DOWN]:
            y += velocity
        if keys[pygame.K_LEFT]:
            x -= velocity
        if keys[pygame.K_RIGHT]:
            x += velocity

        window.fill((0,0,0))
        pygame.draw.rect(window, (255, 0, 0), (x, y, width, height))
        pygame.display.update()

    pygame.quit()

if __name__ == '__main__':
    main()
