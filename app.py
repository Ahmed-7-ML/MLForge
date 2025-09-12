from backend.data import load_data, clean_data
from backend.eda import perform_eda

def main():
    data = load_data('iris.csv')
    cleaned_data = clean_data(data)
    print("Data loaded and cleaned.")

    perform_eda(cleaned_data)

    print("EDA completed.")

main()