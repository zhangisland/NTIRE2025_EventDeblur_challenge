import subprocess


print("Starting Training...")
train_args = [
        "python", "main_train.py",
        "--checkpoints", "./finetune_checkpoints",
        "--config", "finetune_training.yml",
        "--log_dir", "finetune_logs",
        "--method","unet"
    ]
train_process = subprocess.run(train_args, check=True)


if train_process.returncode == 0:
    print("Training completed successfully. Starting Testing...")

    
    test_args = [
        "python", "main_test.py",
        "--checkpoints", "./finetune_checkpoints",
        "--config", "finetune_training.yml",
        "--result_dir", "./finetune_output",
        "--method","unet"
    ]

    
    subprocess.run(test_args, check=True)
else:
    print("Training failed. Testing aborted.")
