import pandas as pd

def calculate_metrics(file_paths):
    scenarios = ["Electrical_Tools", "Mechanical_Tools", "Colors_Shapes_Stationary"]
    scenario_metrics = {}

    combined_df = pd.DataFrame()

    for scenario, file in zip(scenarios, file_paths):
        df = pd.read_csv(file)
        combined_df = pd.concat([combined_df, df], ignore_index=True)

        # Metrics for the current scenario
        total_predictions = len(df)
        correct_predictions = df['1/0'].sum()
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        avg_processing_time = df['processing time'].mean()
        mean_confidence_correct = df.loc[df['1/0'] == 1, 'confidence'].mean()

        scenario_metrics[scenario] = {
            "Accuracy": accuracy,
            "Average Processing Time": avg_processing_time,
            "Mean Confidence (Correct Predictions)": mean_confidence_correct,
        }

    # Overall metrics for the combined model
    overall_metrics = {
        "Accuracy": combined_df['1/0'].sum() / len(combined_df) if len(combined_df) > 0 else 0,
        "Average Processing Time": combined_df['processing time'].mean(),
        "Mean Confidence (Correct Predictions)": combined_df.loc[combined_df['1/0'] == 1, 'confidence'].mean(),
        "Confidence-Weighted Accuracy": (combined_df['confidence'] * combined_df['1/0']).sum() / len(combined_df),
    }

    return scenario_metrics, overall_metrics

# Example Usage
if __name__ == "__main__":
    # Manually provide the file paths
    file_paths = [
        "OwlVit_CSV\\OwlVit_electrical_tools.csv",
        "OwlVit_CSV\\OwlVit_mechanical_tools.csv",
        "OwlVit_CSV\\OwlVit_shapes_stationary.csv"
    ]

    if len(file_paths) == 3:
        scenario_metrics, overall_metrics = calculate_metrics(file_paths)

        print("\nMetrics for Each Scenario:")
        for scenario, metrics in scenario_metrics.items():
            print(f"{scenario}: {metrics}")

        print("\nOverall Metrics:")
        print(overall_metrics)
    else:
        print("Please provide exactly 3 CSV file paths.")
