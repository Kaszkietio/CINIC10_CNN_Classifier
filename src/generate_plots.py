import os
import pandas as pd
import matplotlib.pyplot as plt


def plot_can_training_and_validation_metrics(checkpoints_folder, architecture_groups):
    """
    Plots training and validation metrics for specified groups of architectures and saves the plots as PNG files.

    Args:
        checkpoints_folder (str): Path to the checkpoints folder.
        architecture_groups (dict): Dictionary where keys are group names and values are lists of architecture folder names.
    """
    # Create the 'plots' folder if it doesn't exist
    plots_folder = os.path.join(os.path.dirname(checkpoints_folder), 'plots')
    os.makedirs(plots_folder, exist_ok=True)

    for group_name, architectures in architecture_groups.items():
        # Create a folder for the group inside the 'plots' folder
        group_folder = os.path.join(plots_folder, group_name)
        os.makedirs(group_folder, exist_ok=True)

        # Initialize lists to store data for each architecture in the group
        group_epochs = []
        group_train_loss = []
        group_train_accuracy = []
        group_val_loss = []
        group_val_accuracy = []

        for architecture_folder in architectures:
            architecture_path = os.path.join(checkpoints_folder, architecture_folder)

            # Skip if it's not a directory
            if not os.path.isdir(architecture_path):
                print(f"Warning: {architecture_path} is not a directory. Skipping...")
                continue

            # Paths to the training and validation metrics files
            training_metrics_file = os.path.join(architecture_path, 'training_metrics.csv')
            validation_metrics_file = os.path.join(architecture_path, 'validation_metrics.csv')

            # Check if the files exist
            if not os.path.exists(training_metrics_file) or not os.path.exists(validation_metrics_file):
                print(f"Warning: Metrics files missing in {architecture_path}. Skipping...")
                continue

            # Load the metrics files
            train_metrics = pd.read_csv(training_metrics_file)
            val_metrics = pd.read_csv(validation_metrics_file)

            # Check if required columns exist
            required_columns = {'loss2', 'accuracy'}
            if not required_columns.issubset(train_metrics.columns) or not required_columns.issubset(val_metrics.columns):
                print(f"Error: Metrics files in {architecture_path} must contain the following columns: {required_columns}")
                continue

            # Extract data
            epochs = train_metrics.index + 1  # Assuming the index represents epochs starting from 1
            train_loss = train_metrics['loss2']
            train_accuracy = train_metrics['accuracy']
            val_loss = val_metrics['loss2']
            val_accuracy = val_metrics['accuracy']

            # Append data to group lists
            group_epochs.append(epochs)
            group_train_loss.append(train_loss)
            group_train_accuracy.append(train_accuracy)
            group_val_loss.append(val_loss)
            group_val_accuracy.append(val_accuracy)

        # Plot training loss and accuracy for the group
        fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        for i, epochs in enumerate(group_epochs):
            ax1.plot(epochs, group_train_loss[i], label=f'{architectures[i]} - Training Loss', linestyle='--')
            ax2.plot(epochs, group_train_accuracy[i], label=f'{architectures[i]} - Training Accuracy')
        plt.suptitle(f'{group_name} - Training Metrics')
        ax1.set_xlabel('Epochs')
        ax2.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax2.set_ylabel('Accuracy')
        ax1.legend()
        ax2.legend()
        ax1.grid(True)
        ax2.grid(True)
        training_plot_file = os.path.join(group_folder, f"{group_name}_training_metrics.png")
        plt.tight_layout()
        plt.savefig(training_plot_file)
        plt.close(fig1)
        print(f"Training plot saved to {training_plot_file}")

        # Plot validation loss and accuracy for the group
        fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        for i, epochs in enumerate(group_epochs):
            ax1.plot(epochs, group_val_loss[i], label=f'{architectures[i]} - Validation Loss', linestyle='--')
            ax2.plot(epochs, group_val_accuracy[i], label=f'{architectures[i]} - Validation Accuracy')
        plt.suptitle(f'{group_name} - Validation Metrics')
        ax1.set_xlabel('Epochs')
        ax2.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax2.set_ylabel('Accuracy')
        ax1.legend()
        ax2.legend()
        ax1.grid(True)
        ax2.grid(True)
        validation_plot_file = os.path.join(group_folder, f"{group_name}_validation_metrics.png")
        plt.tight_layout()
        plt.savefig(validation_plot_file)
        plt.close(fig2)
        print(f"Validation plot saved to {validation_plot_file}")


def plot_training_metrics(checkpoints_folder, architecture_groups):
    """
    Plots training metrics for specified groups of architectures and saves the plots as PNG files.

    Args:
        checkpoints_folder (str): Path to the checkpoints folder.
        architecture_groups (dict): Dictionary where keys are group names and values are lists of architecture folder names.
    """
    # Create the 'plots' folder if it doesn't exist
    plots_folder = os.path.join(os.path.dirname(checkpoints_folder), 'plots')
    os.makedirs(plots_folder, exist_ok=True)

    for group_name, architectures in architecture_groups.items():
        # Create a folder for the group inside the 'plots' folder
        group_folder = os.path.join(plots_folder, group_name)
        os.makedirs(group_folder, exist_ok=True)

        # Initialize lists to store data for each architecture in the group
        group_epochs = []
        group_train_loss = []
        group_train_accuracy = []
        group_val_loss = []
        group_val_accuracy = []

        for architecture_folder in architectures:
            architecture_path = os.path.join(checkpoints_folder, architecture_folder)

            # Skip if it's not a directory
            if not os.path.isdir(architecture_path):
                print(f"Warning: {architecture_path} is not a directory. Skipping...")
                continue

            # Path to the metrics.csv file in the current architecture folder
            metrics_file = os.path.join(architecture_path, 'metrics.csv')

            # Check if the file exists
            if not os.path.exists(metrics_file):
                print(f"Warning: {metrics_file} does not exist. Skipping...")
                continue

            # Load the metrics.csv file
            metrics = pd.read_csv(metrics_file)

            # Check if required columns exist
            required_columns = {'loss', 'accuracy', 'val_loss', 'val_accuracy'}
            if not required_columns.issubset(metrics.columns):
                print(f"Error: {metrics_file} must contain the following columns: {required_columns}")
                continue

            # Extract data
            epochs = metrics.index + 1  # Assuming the index represents epochs starting from 1
            train_loss = metrics['loss']
            train_accuracy = metrics['accuracy']
            val_loss = metrics['val_loss']
            val_accuracy = metrics['val_accuracy']

            # Append data to group lists
            group_epochs.append(epochs)
            group_train_loss.append(train_loss)
            group_train_accuracy.append(train_accuracy)
            group_val_loss.append(val_loss)
            group_val_accuracy.append(val_accuracy)

        # Create subplots for the group
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        # Plot training and validation loss for the group
        for i, epochs in enumerate(group_epochs):
            ax1.plot(epochs, group_train_loss[i], label=f'{architectures[i]} - Training Loss', linestyle='--')
            ax1.plot(epochs, group_val_loss[i], label=f'{architectures[i]} - Validation Loss')
        ax1.set_title(f'{group_name} - Loss')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)

        # Plot training and validation accuracy for the group
        for i, epochs in enumerate(group_epochs):
            ax2.plot(epochs, group_train_accuracy[i], label=f'{architectures[i]} - Training Accuracy', linestyle='--')
            ax2.plot(epochs, group_val_accuracy[i], label=f'{architectures[i]} - Validation Accuracy')
        ax2.set_title(f'{group_name} - Accuracy')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)

        # Adjust layout and save the plot
        plt.suptitle(group_name)
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for the suptitle
        plot_file = os.path.join(group_folder, f"{group_name}_metrics.png")
        plt.savefig(plot_file)
        plt.close(fig)  # Close the figure to free memory
        print(f"Plot saved to {plot_file}")

# Example usage
if __name__ == "__main__":
    checkpoints_folder = os.path.join(os.path.dirname(__file__), '..', 'checkpoints')
    # architecture_groups = {
    #     "Dropout regularization": ["AlexNet_dropout_0", "AlexNet_dropout_03", "AlexNet_dropout_05"],
    #     "Weight decay regularization": ["ResNet_wdec_0", "ResNet_wdec_0_01", "ResNet_wdec_0_1",
    #                                     "ResNet_wdec_0_4"],
    #     "Learning rate SGD": ["ResNet_sgd_lr_0_001", "ResNet_sgd_lr_0_01", "ResNet_sgd_lr_0_1"],
    #     "Learning rate AdamW": ["ResNet_adamw_lr_0_001", "ResNet_adamw_lr_0_003",
    #                             "ResNet_adamw_lr_0_01", "ResNet_adamw_lr_0_1"],
    #     "Momentum": ["ResNet_sgd_mom_0_0", "ResNet_sgd_mom_0_5", "ResNet_sgd_mom_0_9",
    #                  "ResNet_sgd_mom_0_99"],
    # }
    # plot_training_metrics(checkpoints_folder, architecture_groups)
    architecture_groups = {
        "Cross Attention Network": ["CAN_lr_0_001", "CAN_lr_0_01", "CAN_lr_0_1"],
    }
    plot_can_training_and_validation_metrics(checkpoints_folder, architecture_groups)