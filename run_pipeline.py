from pipelines.training_pipeline import train_pipeline


if __name__=="__main__":
    print(Client().active_stack.experiment_tracker.get_tracking_uri())
    train_pipeline(data_path=r"D:\Ai projects\customer-satisfaction-mlops-main\venv\data\olist_customers_dataset.csv")

