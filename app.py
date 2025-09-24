import os
from backend.data import load_data, clean_data
from backend.eda import perform_eda


def main():
    dataset_path = "E:\myy projects\Team Project\ZAMESAi-main\ZAMESAi-main\iris.csv"   #### write full path in your pc 

    # 1. Load dataset
    try:
        data = load_data(dataset_path)
        print(f"Data loaded successfully from {dataset_path}")
    except FileNotFoundError:
        print(f"Dataset not found: {dataset_path}")
        return
    except ValueError as e:
        print(f"Error loading dataset: {e}")
        return

    # 2. Clean dataset
    cleaned_data = clean_data(data)
    print("Data cleaned successfully.")

    # 3. Perform EDA 
    perform_eda(cleaned_data)
    print("EDA completed. All plots generated.")

    # 4. Save cleaned dataset for later use
    os.makedirs("outputs", exist_ok=True)
    output_path = os.path.join("outputs", "cleaned_dataset.csv")
    cleaned_data.to_csv(output_path, index=False)
    print(f"Cleaned dataset saved to: {output_path}")


if __name__ == "__main__":
    main()
