# Imports
import csv


# Helper function to export predictions to a CSV or text file
def export(all_predictions: dict, export_to_csv: bool = True, export_folder: str = "results/") -> None:
    export_path = export_folder + 'predictions.csv' if export_to_csv else export_folder + 'predictions.txt'
    if export_to_csv:
        with open(export_path, mode='w') as predictions_file:
            writer = csv.writer(predictions_file)
            writer.writerow(['Image Name', 'Predicted Class', 'Confidence (%)'])
            for name, pred in all_predictions.items():
                writer.writerow([name, pred[0], pred[1]])
    else:
        with open(export_path, mode='w') as predictions_file:
            for name, pred in all_predictions.items():
                predictions_file.write(f"Image Name: {name} | Predicted Class: {pred[0]} | Confidence: {pred[1]: .2f}%\n")
