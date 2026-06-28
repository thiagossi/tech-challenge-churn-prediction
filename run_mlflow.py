import subprocess

subprocess.run(["mlflow", "ui", "--backend-store-uri", "sqlite:///mlruns.db"])
