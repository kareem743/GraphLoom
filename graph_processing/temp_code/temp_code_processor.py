def printfun():
    print("Hello from printfun!")

def loop():
    #This function loops 3 times and calls printfun.
    for i in range(3):
        printfun()

def run_loops(num_times):
    #Runs the loop function multiple times.
    print(f"Running the loop {num_times} times.")
    for _ in range(num_times):
        loop()
    return f"Completed {num_times} loop runs."