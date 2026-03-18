import torch
from task import TabularMLP, load_data, train, test

def main():
    # 1. Cấu hình thử nghiệm
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 32
    LEARNING_RATE = 0.01
    EPOCHS = 1
    PARTITION_ID = 0  # Chạy thử với partition đầu tiên (Dataset 1)
    NUM_PARTITIONS = 5

    print(f"--- Đang chạy thử trên thiết bị: {DEVICE} ---")

    # 2. Thử load dữ liệu
    print("\n[Step 1] Đang load dữ liệu...")

    try:
        trainloader, testloader = load_data(
            partition_id=PARTITION_ID, 
            num_partitions=NUM_PARTITIONS, 
            batch_size=BATCH_SIZE
        )
        print(f"✅ Load thành công!")
        print(f" - Số lượng mẫu training: {len(trainloader.dataset)}")
        print(f" - Số lượng mẫu testing: {len(testloader.dataset)}")
    except Exception as e:
        print(f"❌ Lỗi khi load dữ liệu: {e}")
        return

    # 3. Khởi tạo Model
    print("\n[Step 2] Khởi tạo model (TabularMLP)...")
    net = TabularMLP(input_dim=4).to(DEVICE)
    print("✅ Model đã sẵn sàng.")

    # 4. Chạy thử training
    print(f"\n[Step 3] Chạy thử training ({EPOCHS} epoch)...")
    try:
        avg_loss = train(net, trainloader, EPOCHS, LEARNING_RATE, DEVICE)
        print(f"✅ Training hoàn tất. Loss trung bình: {avg_loss:.4f}")
    except Exception as e:
        print(f"❌ Lỗi khi training: {e}")
        return

    # 5. Chạy thử evaluation
    print("\n[Step 4] Chạy thử evaluation...")
    try:
        loss, accuracy = test(net, testloader, DEVICE)
        print(f"✅ Eval hoàn tất.")
        print(f" - Accuracy: {accuracy:.4f}")
        print(f" - Loss: {loss:.4f}")
    except Exception as e:
        print(f"❌ Lỗi khi evaluation: {e}")

    print("\n--- Tất cả các bước chạy thử đã hoàn tất thành công! ---")

if __name__ == "__main__":
    main()
