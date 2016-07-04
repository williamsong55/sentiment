from Decider import Decider


decider = Decider("Naive")
while True:
    words = input("Input sentence:\n")
    decider.predict(words)


