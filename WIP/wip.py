import threading

def printit():
    threading.Timer(3.0, printit).start()
    print("Hello, World!")

printit()