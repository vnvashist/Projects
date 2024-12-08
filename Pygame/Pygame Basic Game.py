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

    isJump = False
    jumpCount = 10

    run = True
    while run:
        pygame.time.delay(25)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT] and x > 0:
            x -= velocity
        if keys[pygame.K_RIGHT] and x < 500 - width - velocity:
            x += velocity
        if not isJump:
            if keys[pygame.K_UP] and y > velocity:
                y -= velocity
            if keys[pygame.K_DOWN] and y < 500 - height - velocity:
                y += velocity
            if keys[pygame.K_SPACE]:
                isJump = True
        else:
            if jumpCount >= -10:
                neg = 1
                if jumpCount < 0:
                    neg = -1
                y -= (jumpCount ** 2) * 0.5 * neg
                jumpCount -= 1
            else:
                isJump = False
                jumpCount = 10

        window.fill((0,0,0))
        pygame.draw.rect(window, (255, 0, 0), (x, y, width, height))
        pygame.display.update()

    pygame.quit()

if __name__ == '__main__':
    main()
