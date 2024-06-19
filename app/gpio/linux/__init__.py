import gpiod

CONSUMER = "radar"
EXISTENCE_LINE = 6
LED_LINE = 203
CHIPNAME = "gpiochip0"

# please note that this code is for v1.6.x of libgpiod
# https://git.kernel.org/pub/scm/libs/libgpiod/libgpiod.git
# there's big changes in v2.x


class GPIO:

    def __init__(self):
        self.chip = gpiod.Chip(CHIPNAME)
        lines = self.chip.get_lines([EXISTENCE_LINE, LED_LINE])
        lines.request(consumer=CONSUMER, type=gpiod.LINE_REQ_DIR_OUT)
        lines.set_values([0, 0])
        self.lines = lines

    def low(self):
        self.lines.set_values([0, 0])

    def high(self):
        self.lines.set_values([1, 1])


def main():
    from time import sleep
    gpio = GPIO()
    while True:
        gpio.high()
        sleep(1)
        gpio.low()
        sleep(1)


if __name__ == "__main__":
    main()
