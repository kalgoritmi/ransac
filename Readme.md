## Installation

You need at least python 3.8+ (a lot of walruses :=).

You need to have make and python installed, e.g.:

```
apt/yum install make 
```

The init target in Makefile creates, activates and installs requirements to
a virtual environment.

```
make init 
```

## Running the demo

The main entry point is `app.py`, you can run the app either by:
```
make run
```
or

```
python3 app.py
```

or

```
chmod +x app.py
./app.py
```

Then head to localhost:8050

## Cleaning

```
make clean
```
