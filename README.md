# using doppler radar sensor to detect human presence

a naive way

```bash
apt install -y \
    python3-pydantic \
    python3-serial \
    python3-libgpiod \
    gpiod \
    python3-loguru \
    python3-numpy \
    python3-click
```

```bash
python -m pip install \
    gpiod \
    pydantic \
    pyserial \
    scikit-fuzzy
```

```bash
# PA6  6
# PG11 203
gpioset gpiochip0 6=1
gpioset gpiochip0 203=1
```
