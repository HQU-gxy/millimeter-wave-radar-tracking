# using doppler radar sensor to detect human presence

a naive way

```bash
apt install -y \
    git \
    python3-pydantic \
    python3-serial \
    python3-numpy \
    python3-matplotlib \
    python3-sklearn \
    python3-libgpiod \
    gpiod \
    python3-loguru \
    python3-numpy \
    python3-click
```

```bash
python3 -m pip install -U \
    scikit-fuzzy --break-system-packages
python3 -m pip install -U \
    anyio  --break-system-packages
```

```bash
# PA6  6
# PG11 203
gpioset gpiochip0 6=1
gpioset gpiochip0 203=1
```

```bash
HOST=192.168.2.226
PORT=7890
export http_proxy=
export https_proxy=
export all_proxy=
export no_proxy=
export HTTP_PROXY=http://$HOST:$PORT
export HTTPS_PROXY=http://$HOST:$PORT
export NO_PROXY=127.0.0.1,127.0.0.0/8,192.168.0.0/16,10.0.0.0/8,172.16.0.0/12
```

| Primary tables    | Object type | Type of    | Comments                                                      |
| ----------------- | ----------- | ---------- | ------------------------------------------------------------- |
| Discretes Input   | Single bit  | Read-Only  | This type of data can be provided by an I/O system.           |
| Coils             | Single bit  | Read-Write | This type of data can be alterable by an application program. |
| Input Registers   | 16-bit word | Read-Only  | This type of data can be provided by an I/O system            |
| Holding Registers | 16-bit word | Read-Write | This type of data can be alterable by an application program. |

