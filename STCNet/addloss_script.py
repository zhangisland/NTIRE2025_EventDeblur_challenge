import subprocess

# 运行训练
print("Starting Training...")
train_args = [
        "python", "main_train.py",
        "--checkpoints", "./addloss_checkpoints",
        "--config", "training.yml",
        "--log_dir", "addloss_logs",
        "--method","unet"
    ]
train_process = subprocess.run(train_args, check=True)

# 只有当训练成功时才执行测试
if train_process.returncode == 0:
    print("Training completed successfully. Starting Testing...")

    # 定义测试脚本的参数
    test_args = [
        "python", "main_test.py",
        "--checkpoints", "./addloss_checkpoints",
        "--config", "training.yml",
        "--result_dir", "./addloss_output",
        "--method","unet"
    ]

    # 运行测试脚本
    subprocess.run(test_args, check=True)
else:
    print("Training failed. Testing aborted.")
