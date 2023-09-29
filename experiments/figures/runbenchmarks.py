import subprocess

# Path to the Python script to run
script_path = "benchmarks.py"

# Loop through the parameter values
#for modeltrain in [True, False]:
for modeltrain in [True]:
    for model_choice in range(1, 6):  # 1 to 5 inclusive
        # for subvolume_size in range(1, 3):  # 1 to 2 inclusive
        for subvolume_size in range(2, 3):  # 1 to 2 inclusive

            # Convert boolean to string
            modeltrain_str = "True" if modeltrain else "False"

            # Call the script with subprocess
            try:
                if modeltrain:
                    subprocess.run(["python", script_path,
                                "--modeltrain", modeltrain_str,
                                "--model_choice", str(model_choice),
                                "--subvolume_size", str(subvolume_size)])
                else:
                    subprocess.run(["python", script_path,
                        "--model_choice", str(model_choice),
                        "--subvolume_size", str(subvolume_size)])

            except Exception as e:
                import traceback 
                traceback.print_exc()
                import time
                time.sleep(10)
            # Note: Replace "python" with "python3" if that's the version you're using.
