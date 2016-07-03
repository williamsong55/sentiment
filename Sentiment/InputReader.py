from Decider import Decider


decider = Decider("MaxEnt")
while True:

    cmd = input("Please enter a command\n")

    if cmd=="predict":
        words = input("Input sentence:\n")
        decider.predict(words)

    elif cmd=="load":
        decider.load()

    elif cmd=="build":
        decider.build_model()

    elif cmd == "feat":
        decider.features()
    elif cmd=="q":
        break
    else:
        print("predict - load - build - feat - q")

