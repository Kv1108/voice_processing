import subprocess

# Start A.py and B.py
process_a = subprocess.Popen(["python", "CCTV.py"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
process_b = subprocess.Popen(["python", "record.py"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

# Optional: Capture and print outputs from A.py and B.py
print("Running...")

try:
    # Wait for both processes to finish (if needed)
    output_a, error_a = process_a.communicate()
    output_b, error_b = process_b.communicate()

    # Print outputs (optional)
    if output_a:
        print(f"CCTV.py Output:\n{output_a.decode()}")
    if error_a:
        print(f"CCTV.py Error:\n{error_a.decode()}")

    if output_b:
        print(f"record.py Output:\n{output_b.decode()}")
    if error_b:
        print(f"record.py Error:\n{error_b.decode()}")

except KeyboardInterrupt:
    print("Terminating processes...")
    process_a.terminate()
    process_b.terminate()
    print("Processes terminated.")
