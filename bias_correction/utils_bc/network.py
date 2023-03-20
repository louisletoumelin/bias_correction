import os


def detect_network():

    cwd = os.getcwd()
    labia_key_detected = "mrmn" in cwd
    probably_on_docker = not "letoumelinl" in cwd

    if labia_key_detected or probably_on_docker:
        print("\nWorking on labia")
        return "labia"
    else:
        print("\nWorking on local")
        return "local"
